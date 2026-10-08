#include <CL/cl.h>
#include <jni.h>
#include <dlfcn.h>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(cl_int status, const char* name) {
    if (status != CL_SUCCESS) throw std::runtime_error(std::string(name) + " failed: " + std::to_string(status));
}
struct API {
    void* library = nullptr;
#define FUNCTIONS(F) \
    F(clGetPlatformIDs) F(clGetDeviceIDs) F(clGetDeviceInfo) F(clCreateContext) F(clCreateCommandQueue) \
    F(clCreateProgramWithSource) F(clBuildProgram) F(clGetProgramBuildInfo) F(clCreateKernel) \
    F(clCreateBuffer) F(clSetKernelArg) F(clEnqueueWriteBuffer) F(clEnqueueNDRangeKernel) \
    F(clEnqueueReadBuffer) F(clReleaseMemObject) F(clReleaseKernel) F(clReleaseProgram) \
    F(clReleaseCommandQueue) F(clReleaseContext)
#define DECLARE(name) decltype(&name) name = nullptr;
    FUNCTIONS(DECLARE)
#undef DECLARE
    API() {
        library = dlopen("libOpenCL.so", RTLD_NOW | RTLD_LOCAL);
        if (!library) throw std::runtime_error("OpenCL library unavailable");
#define LOAD(name) name = reinterpret_cast<decltype(name)>(dlsym(library, #name)); if (!name) { dlclose(library); library=nullptr; throw std::runtime_error("Missing " #name); }
        FUNCTIONS(LOAD)
#undef LOAD
    }
    ~API() { if (library) dlclose(library); }
};
const char* source = R"CLC(
float channel(__global const uchar* pixels, int row, int x, int y, int c, int w, int h) {
    x=clamp(x,0,w-1); y=clamp(y,0,h-1);
    return pixels[y*row+x*4+c];
}
__kernel void prepare(__global const uchar* pixels, __global float* result,
                      int row, int w, int h, int nw, int nh) {
    int i=get_global_id(0); if(i>=800*384*3) return;
    int c=i%3, x=(i/3)%384-(384-nw)/2, y=i/(384*3)-(800-nh)/2;
    float value=c==0?124.0f:(c==1?116.0f:104.0f);
    if(x>=0 && x<nw && y>=0 && y<nh) {
        float sx=(float)w/nw, sy=(float)h/nh;
        if(nw<w || nh<h) {
            float x0=x*sx, x1=(x+1)*sx, y0=y*sy, y1=(y+1)*sy;
            float sum=0.0f;
            for(int yy=(int)floor(y0); yy<(int)ceil(y1); yy++)
                for(int xx=(int)floor(x0); xx<(int)ceil(x1); xx++) {
                    float wx=fmax(0.0f,fmin(x1,xx+1.0f)-fmax(x0,(float)xx));
                    float wy=fmax(0.0f,fmin(y1,yy+1.0f)-fmax(y0,(float)yy));
                    sum+=channel(pixels,row,xx,yy,c,w,h)*wx*wy;
                }
            value=clamp(rint(sum/(sx*sy)),0.0f,255.0f);
        } else {
            float fx=(x+0.5f)*sx-0.5f, fy=(y+0.5f)*sy-0.5f;
            int xx=(int)floor(fx), yy=(int)floor(fy);
            float dx=fx-xx, dy=fy-yy;
            float top=mix(channel(pixels,row,xx,yy,c,w,h),channel(pixels,row,xx+1,yy,c,w,h),dx);
            float bottom=mix(channel(pixels,row,xx,yy+1,c,w,h),channel(pixels,row,xx+1,yy+1,c,w,h),dx);
            value=clamp(rint(mix(top,bottom,dy)),0.0f,255.0f);
        }
    }
    float mean=c==0?0.485f:(c==1?0.456f:0.406f);
    float std=c==0?0.229f:(c==1?0.224f:0.225f);
    result[i]=(value/255.0f-mean)/std;
}
)CLC";
struct Pipeline {
    API api;
    cl_context context=nullptr;
    cl_command_queue queue=nullptr;
    cl_program program=nullptr;
    cl_kernel kernel=nullptr;
    cl_mem input=nullptr, output=nullptr;
    size_t capacity=0;
    std::string driver;
    std::vector<unsigned char> packed;
    Pipeline() {
        try {
            cl_uint count=0; check(api.clGetPlatformIDs(0,nullptr,&count),"Platforms");
            std::vector<cl_platform_id> platforms(count);
            check(api.clGetPlatformIDs(count,platforms.data(),nullptr),"Platforms");
            cl_device_id device=nullptr;
            for (auto platform:platforms) {
                if (api.clGetDeviceIDs(platform,CL_DEVICE_TYPE_GPU,1,&device,nullptr)==CL_SUCCESS) break;
                device=nullptr;
            }
            if(!device) throw std::runtime_error("No OpenCL GPU");
            char driverVersion[256]{},deviceName[256]{};
            check(api.clGetDeviceInfo(device,CL_DRIVER_VERSION,sizeof(driverVersion),driverVersion,nullptr),"Driver version");
            check(api.clGetDeviceInfo(device,CL_DEVICE_NAME,sizeof(deviceName),deviceName,nullptr),"Device name");
            driver=std::string(deviceName)+":"+driverVersion;
            cl_int error=0;
            context=api.clCreateContext(nullptr,1,&device,nullptr,nullptr,&error); check(error,"Context");
            queue=api.clCreateCommandQueue(context,device,0,&error); check(error,"Queue");
            program=api.clCreateProgramWithSource(context,1,&source,nullptr,&error); check(error,"Program");
            error=api.clBuildProgram(program,1,&device,"-cl-std=CL1.2",nullptr,nullptr);
            if(error!=CL_SUCCESS) {
                char log[8192]{}; api.clGetProgramBuildInfo(program,device,CL_PROGRAM_BUILD_LOG,sizeof(log)-1,log,nullptr);
                throw std::runtime_error(std::string("OpenCL build: ")+log);
            }
            kernel=api.clCreateKernel(program,"prepare",&error); check(error,"Kernel");
            output=api.clCreateBuffer(context,CL_MEM_WRITE_ONLY,800*384*3*sizeof(float),nullptr,&error); check(error,"Output");
        } catch(...) { release(); throw; }
    }
    void release() {
        if(input) api.clReleaseMemObject(input);
        if(output) api.clReleaseMemObject(output);
        if(kernel) api.clReleaseKernel(kernel);
        if(program) api.clReleaseProgram(program);
        if(queue) api.clReleaseCommandQueue(queue);
        if(context) api.clReleaseContext(context);
        input=nullptr; output=nullptr; kernel=nullptr; program=nullptr; queue=nullptr; context=nullptr;
    }
    ~Pipeline() { release(); }
    void ingest(const unsigned char* bytes, size_t available, int stride, int left, int top, int w, int h) {
        if(w<=0 || h<=0 || w>16384 || h>16384 || static_cast<size_t>(w)*h*4>128*1024*1024 || stride < (left+w)*4 || left<0 || top<0 ||
           available < static_cast<size_t>(top+h-1)*stride+(left+w)*4) throw std::runtime_error("Invalid RGBA plane bounds");
        size_t needed=static_cast<size_t>(w)*h*4;
        if(needed>capacity) {
            if(input) { api.clReleaseMemObject(input); input=nullptr; }
            cl_int error=0; input=api.clCreateBuffer(context,CL_MEM_READ_ONLY,needed,nullptr,&error); check(error,"Input");
            capacity=needed;
        }
        packed.resize(needed);
        for(int y=0;y<h;y++) std::memcpy(packed.data()+static_cast<size_t>(y)*w*4,bytes+static_cast<size_t>(top+y)*stride+left*4,w*4);
        // Blocking upload: ImageReader's borrowed plane may close after this call.
        check(api.clEnqueueWriteBuffer(queue,input,CL_TRUE,0,needed,packed.data(),0,nullptr,nullptr),"Upload");
    }
    void prepare(float* result,int w,int h,int nw,int nh) {
        int row=w*4;
        check(api.clSetKernelArg(kernel,0,sizeof(input),&input),"Arg");
        check(api.clSetKernelArg(kernel,1,sizeof(output),&output),"Arg");
        int args[]={row,w,h,nw,nh};
        for(int i=0;i<5;i++) check(api.clSetKernelArg(kernel,i+2,sizeof(int),&args[i]),"Arg");
        size_t global=800*384*3;
        check(api.clEnqueueNDRangeKernel(queue,kernel,1,nullptr,&global,nullptr,0,nullptr,nullptr),"Dispatch");
        check(api.clEnqueueReadBuffer(queue,output,CL_TRUE,0,global*sizeof(float),result,0,nullptr,nullptr),"Readback");
    }
};
void fail(JNIEnv* env,const std::exception& error) { env->ThrowNew(env->FindClass("java/lang/IllegalStateException"),error.what()); }
}
extern "C" JNIEXPORT jlong JNICALL Java_ro_ubb_uicollector_OpenClPreprocessor_create(JNIEnv* env,jobject) {
    try { return reinterpret_cast<jlong>(new Pipeline()); } catch(const std::exception& error) { fail(env,error); return 0; }
}
extern "C" JNIEXPORT void JNICALL Java_ro_ubb_uicollector_OpenClPreprocessor_ingestNative(JNIEnv* env,jobject,jlong handle,jobject bytes,jint row,jint left,jint top,jint w,jint h) {
    try {
        auto* data=static_cast<unsigned char*>(env->GetDirectBufferAddress(bytes));
        if(!handle || !data) throw std::runtime_error("Missing direct RGBA buffer");
        reinterpret_cast<Pipeline*>(handle)->ingest(data,env->GetDirectBufferCapacity(bytes),row,left,top,w,h);
    } catch(const std::exception& error) { fail(env,error); }
}
extern "C" JNIEXPORT void JNICALL Java_ro_ubb_uicollector_OpenClPreprocessor_prepareNative(JNIEnv* env,jobject,jlong handle,jobject bytes,jint w,jint h,jint nw,jint nh) {
    try {
        auto* data=static_cast<float*>(env->GetDirectBufferAddress(bytes));
        if(!handle || !data || env->GetDirectBufferCapacity(bytes)<800*384*3*4) throw std::runtime_error("Missing direct output buffer");
        reinterpret_cast<Pipeline*>(handle)->prepare(data,w,h,nw,nh);
    } catch(const std::exception& error) { fail(env,error); }
}
extern "C" JNIEXPORT void JNICALL Java_ro_ubb_uicollector_OpenClPreprocessor_destroy(JNIEnv*,jobject,jlong handle) { delete reinterpret_cast<Pipeline*>(handle); }

extern "C" JNIEXPORT jstring JNICALL Java_ro_ubb_uicollector_OpenClPreprocessor_driverNative(JNIEnv* env,jobject,jlong handle) {
    return env->NewStringUTF(reinterpret_cast<Pipeline*>(handle)->driver.c_str());
}
