#define CL_TARGET_OPENCL_VERSION 300
#define SDL_MAIN_HANDLED
#include <CL/cl.h>
#include <SDL2/SDL.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static const int WIDTH = 1280;
static const int HEIGHT = 720;

static const char* kKernelSource = R"CLC(
typedef struct {
    float3 center;
    float radius;
    float3 color;
    float emissive;
} Sphere;

float intersect_sphere(float3 ro, float3 rd, Sphere s) {
    float3 oc = ro - s.center;
    float b = dot(oc, rd);
    float c = dot(oc, oc) - s.radius * s.radius;
    float h = b * b - c;
    if (h < 0.0f) return -1.0f;
    h = sqrt(h);
    float t = -b - h;
    if (t > 0.001f) return t;
    t = -b + h;
    return (t > 0.001f) ? t : -1.0f;
}

float3 shade(float3 ro, float3 rd, __global const Sphere* spheres, int sphereCount) {
    float tMin = 1e20f;
    int hit = -1;

    for (int i = 0; i < sphereCount; ++i) {
        float t = intersect_sphere(ro, rd, spheres[i]);
        if (t > 0.0f && t < tMin) {
            tMin = t;
            hit = i;
        }
    }

    float3 bgTop = (float3)(0.01f, 0.02f, 0.06f);
    float3 bgBottom = (float3)(0.0f, 0.0f, 0.0f);
    float blend = clamp(0.5f * (rd.y + 1.0f), 0.0f, 1.0f);
    float3 background = mix(bgBottom, bgTop, blend);

    if (hit < 0) {
        float noise = sin(dot((float2)(rd.x, rd.y), (float2)(12.9898f, 78.233f))) * 43758.5453f;
        float stars = noise - floor(noise);
        if (stars > 0.9975f) {
            background += (float3)(0.8f, 0.8f, 0.8f) * (stars - 0.9975f) * 300.0f;
        }
        return clamp(background, 0.0f, 1.0f);
    }

    Sphere s = spheres[hit];
    float3 p = ro + rd * tMin;
    float3 n = normalize(p - s.center);

    float3 lightDir = normalize(spheres[0].center - p);
    float diff = fmax(dot(n, lightDir), 0.0f);

    float distToSun = length(spheres[0].center - p);
    float attenuation = 1.0f / (1.0f + 0.03f * distToSun * distToSun);

    float3 ambient = 0.08f * s.color;
    float3 direct = diff * attenuation * s.color * 4.0f;
    float3 emissive = s.color * s.emissive;

    return clamp(ambient + direct + emissive + background * 0.03f, 0.0f, 1.0f);
}

__kernel void render(
    __global uchar4* pixels,
    __global const Sphere* spheres,
    const int sphereCount,
    const int width,
    const int height,
    const float time
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x >= width || y >= height) return;

    float2 uv = (float2)(
        ((float)x + 0.5f) / (float)width,
        ((float)y + 0.5f) / (float)height
    );
    uv = uv * 2.0f - 1.0f;
    uv.x *= (float)width / (float)height;

    float3 camPos = (float3)(0.0f, 8.0f, 30.0f);
    float orbit = time * 0.12f;
    camPos.x = sin(orbit) * 8.0f;
    camPos.z = 30.0f + cos(orbit) * 4.0f;

    float3 target = (float3)(0.0f, 0.0f, 0.0f);
    float3 forward = normalize(target - camPos);
    float3 right = normalize(cross((float3)(0.0f, 1.0f, 0.0f), forward));
    float3 up = cross(forward, right);

    float fov = 1.2f;
    float3 rd = normalize(forward + uv.x * right * fov + uv.y * up * fov);

    float3 color = shade(camPos, rd, spheres, sphereCount);

    int idx = y * width + x;
    pixels[idx] = (uchar4)(
        (uchar)(255.0f * clamp(color.x, 0.0f, 1.0f)),
        (uchar)(255.0f * clamp(color.y, 0.0f, 1.0f)),
        (uchar)(255.0f * clamp(color.z, 0.0f, 1.0f)),
        255
    );
}
)CLC";

struct alignas(16) Float3Pad {
    float x, y, z, w;
};

struct alignas(16) SphereHost {
    Float3Pad center;
    float radius;
    float pad0[3];
    Float3Pad color;
    float emissive;
    float pad1[3];
};

static void checkErr(cl_int err, const std::string& what) {
    if (err != CL_SUCCESS) {
        std::cerr << "OpenCL error (" << err << ") at: " << what << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

static cl_device_id pickDevice(cl_platform_id& outPlatform) {
    cl_uint platformCount = 0;
    checkErr(clGetPlatformIDs(0, nullptr, &platformCount), "clGetPlatformIDs count");
    if (platformCount == 0) {
        std::cerr << "No OpenCL platforms found." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::vector<cl_platform_id> platforms(platformCount);
    checkErr(clGetPlatformIDs(platformCount, platforms.data(), nullptr), "clGetPlatformIDs list");

    for (auto platform : platforms) {
        cl_uint deviceCount = 0;
        cl_int err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &deviceCount);
        if (err == CL_SUCCESS && deviceCount > 0) {
            std::vector<cl_device_id> devices(deviceCount);
            checkErr(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, deviceCount, devices.data(), nullptr), "clGetDeviceIDs GPU");
            outPlatform = platform;
            return devices[0];
        }
    }

    for (auto platform : platforms) {
        cl_uint deviceCount = 0;
        cl_int err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, nullptr, &deviceCount);
        if (err == CL_SUCCESS && deviceCount > 0) {
            std::vector<cl_device_id> devices(deviceCount);
            checkErr(clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, deviceCount, devices.data(), nullptr), "clGetDeviceIDs CPU");
            outPlatform = platform;
            return devices[0];
        }
    }

    std::cerr << "No suitable OpenCL device found." << std::endl;
    std::exit(EXIT_FAILURE);
}

static std::string getBuildLog(cl_program program, cl_device_id device) {
    size_t logSize = 0;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
    std::string log(logSize, '\0');
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
    return log;
}

static SphereHost makeSphere(float x, float y, float z, float radius, float r, float g, float b, float emissive = 0.0f) {
    SphereHost s{};
    s.center = {x, y, z, 0.0f};
    s.radius = radius;
    s.color = {r, g, b, 0.0f};
    s.emissive = emissive;
    return s;
}

int main() {
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::cerr << "SDL_Init failed: " << SDL_GetError() << std::endl;
        return EXIT_FAILURE;
    }

    SDL_Window* window = SDL_CreateWindow(
        "OpenCL 3D Solar System",
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        WIDTH,
        HEIGHT,
        SDL_WINDOW_SHOWN
    );
    if (!window) {
        std::cerr << "SDL_CreateWindow failed: " << SDL_GetError() << std::endl;
        SDL_Quit();
        return EXIT_FAILURE;
    }

    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer) {
        std::cerr << "SDL_CreateRenderer failed: " << SDL_GetError() << std::endl;
        SDL_DestroyWindow(window);
        SDL_Quit();
        return EXIT_FAILURE;
    }

    SDL_Texture* texture = SDL_CreateTexture(
        renderer,
        SDL_PIXELFORMAT_RGBA8888,
        SDL_TEXTUREACCESS_STREAMING,
        WIDTH,
        HEIGHT
    );
    if (!texture) {
        std::cerr << "SDL_CreateTexture failed: " << SDL_GetError() << std::endl;
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return EXIT_FAILURE;
    }

    cl_platform_id platform = nullptr;
    cl_device_id device = pickDevice(platform);

    char deviceName[256] = {};
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
    std::cout << "Using OpenCL device: " << deviceName << std::endl;

    cl_int err = CL_SUCCESS;
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    checkErr(err, "clCreateContext");

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    checkErr(err, "clCreateCommandQueue");

    const char* src = kKernelSource;
    size_t srcLen = std::strlen(kKernelSource);
    cl_program program = clCreateProgramWithSource(context, 1, &src, &srcLen, &err);
    checkErr(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Kernel build failed:\n" << getBuildLog(program, device) << std::endl;
        checkErr(err, "clBuildProgram");
    }

    cl_kernel kernel = clCreateKernel(program, "render", &err);
    checkErr(err, "clCreateKernel");

    std::vector<uint32_t> pixels(WIDTH * HEIGHT, 0);
    cl_mem pixelBuffer = clCreateBuffer(
        context,
        CL_MEM_WRITE_ONLY,
        pixels.size() * sizeof(uint32_t),
        nullptr,
        &err
    );
    checkErr(err, "clCreateBuffer pixelBuffer");

    constexpr int kSphereCount = 9;
    std::vector<SphereHost> spheres(kSphereCount);
    cl_mem sphereBuffer = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY,
        spheres.size() * sizeof(SphereHost),
        nullptr,
        &err
    );
    checkErr(err, "clCreateBuffer sphereBuffer");

    bool running = true;
    uint32_t startTicks = SDL_GetTicks();

    while (running) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = false;
            }
            if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE) {
                running = false;
            }
        }

        float time = (SDL_GetTicks() - startTicks) * 0.001f;

        spheres[0] = makeSphere(0.0f, 0.0f, 0.0f, 2.8f, 1.00f, 0.84f, 0.25f, 2.5f); // Sun

        const float tilt = 0.18f;
        auto orbitPos = [&](float radius, float speed, float phase) {
            float a = time * speed + phase;
            float x = std::cos(a) * radius;
            float z = std::sin(a) * radius;
            float y = std::sin(a * 0.6f) * radius * tilt;
            return Float3Pad{x, y, z, 0.0f};
        };

        spheres[1] = makeSphere(0, 0, 0, 0.35f, 0.65f, 0.62f, 0.60f);  // Mercury
        spheres[1].center = orbitPos(5.0f, 1.8f, 0.0f);

        spheres[2] = makeSphere(0, 0, 0, 0.50f, 0.90f, 0.76f, 0.54f); // Venus
        spheres[2].center = orbitPos(7.2f, 1.3f, 1.1f);

        spheres[3] = makeSphere(0, 0, 0, 0.55f, 0.18f, 0.45f, 0.95f);;  // Earth
        spheres[3].center = orbitPos(10.0f, 1.0f, 2.0f);

        spheres[4] = makeSphere(0, 0, 0, 0.42f, 0.80f, 0.42f, 0.25f);  // Mars
        spheres[4].center = orbitPos(13.0f, 0.8f, 0.3f);

        spheres[5] = makeSphere(0, 0, 0, 1.20f, 0.82f, 0.70f, 0.50f); // Jupiter
        spheres[5].center = orbitPos(17.0f, 0.55f, 1.8f);

        spheres[6] = makeSphere(0, 0, 0, 1.00f, 0.88f, 0.80f, 0.62f); // Saturn
        spheres[6].center = orbitPos(21.0f, 0.42f, 2.6f);

        spheres[7] = makeSphere(0, 0, 0, 0.78f, 0.58f, 0.82f, 0.88f); // Uranus
        spheres[7].center = orbitPos(25.0f, 0.32f, 0.7f);

        spheres[8] = makeSphere(0, 0, 0, 0.76f, 0.12f, 0.22f, 0.72f); // Neptune
        spheres[8].center = orbitPos(29.0f, 0.26f, 2.9f);

        checkErr(clEnqueueWriteBuffer(queue, sphereBuffer, CL_TRUE, 0,
            spheres.size() * sizeof(SphereHost), spheres.data(), 0, nullptr, nullptr),
            "clEnqueueWriteBuffer sphereBuffer");

        checkErr(clSetKernelArg(kernel, 0, sizeof(cl_mem), &pixelBuffer), "clSetKernelArg 0");
        checkErr(clSetKernelArg(kernel, 1, sizeof(cl_mem), &sphereBuffer), "clSetKernelArg 1");
        checkErr(clSetKernelArg(kernel, 2, sizeof(int), &kSphereCount), "clSetKernelArg 2");
        checkErr(clSetKernelArg(kernel, 3, sizeof(int), &WIDTH), "clSetKernelArg 3");
        checkErr(clSetKernelArg(kernel, 4, sizeof(int), &HEIGHT), "clSetKernelArg 4");
        checkErr(clSetKernelArg(kernel, 5, sizeof(float), &time), "clSetKernelArg 5");

        size_t globalSize[2] = {static_cast<size_t>(WIDTH), static_cast<size_t>(HEIGHT)};
        checkErr(clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalSize, nullptr, 0, nullptr, nullptr),
            "clEnqueueNDRangeKernel");
        checkErr(clFinish(queue), "clFinish");

        checkErr(clEnqueueReadBuffer(queue, pixelBuffer, CL_TRUE, 0,
            pixels.size() * sizeof(uint32_t), pixels.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer pixelBuffer");

        SDL_UpdateTexture(texture, nullptr, pixels.data(), WIDTH * sizeof(uint32_t));
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, nullptr, nullptr);
        SDL_RenderPresent(renderer);
    }

    clReleaseMemObject(sphereBuffer);
    clReleaseMemObject(pixelBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}