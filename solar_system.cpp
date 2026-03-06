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

typedef struct {
    float3 normal;
    float t;
    float3 color;
    float emissive;
    int hitType; // 0 none, 1 sphere, 2 ring
} HitInfo;

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

float intersect_ring_xz(float3 ro, float3 rd, float3 center, float innerR, float outerR) {
    if (fabs(rd.y) < 1e-5f) return -1.0f;
    float t = (center.y - ro.y) / rd.y;
    if (t <= 0.001f) return -1.0f;
    float3 p = ro + rd * t;
    float2 d = (float2)(p.x - center.x, p.z - center.z);
    float r = length(d);
    if (r >= innerR && r <= outerR) return t;
    return -1.0f;
}

float ring_mask(float radius, float innerR, float outerR) {
    float norm = (radius - innerR) / (outerR - innerR);
    float bands = 0.55f + 0.45f * sin(norm * 80.0f);
    float fade = smoothstep(innerR, innerR + 0.12f, radius) * (1.0f - smoothstep(outerR - 0.12f, outerR, radius));
    return clamp(bands * fade, 0.0f, 1.0f);
}

float star_noise(float2 p) {
    float n = sin(dot(p, (float2)(12.9898f, 78.233f))) * 43758.5453f;
    return n - floor(n);
}

HitInfo trace_scene(float3 ro, float3 rd, __global const Sphere* spheres, int sphereCount) {
    HitInfo hit;
    hit.t = 1e20f;
    hit.hitType = 0;
    hit.color = (float3)(0.0f, 0.0f, 0.0f);
    hit.normal = (float3)(0.0f, 1.0f, 0.0f);
    hit.emissive = 0.0f;

    for (int i = 0; i < sphereCount; ++i) {
        float t = intersect_sphere(ro, rd, spheres[i]);
        if (t > 0.0f && t < hit.t) {
            float3 p = ro + rd * t;
            hit.t = t;
            hit.hitType = 1;
            hit.normal = normalize(p - spheres[i].center);
            hit.color = spheres[i].color;
            hit.emissive = spheres[i].emissive;
        }
    }

    // Orbit rings aligned with the planets around the sun in the XZ plane.
    float3 sunCenter = spheres[0].center;
    for (int i = 1; i < sphereCount; ++i) {
        float orbitRadius = length((float2)(spheres[i].center.x - sunCenter.x, spheres[i].center.z - sunCenter.z));
        float t = intersect_ring_xz(ro, rd, sunCenter, orbitRadius - 0.05f, orbitRadius + 0.05f);
        if (t > 0.0f && t < hit.t) {
            hit.t = t;
            hit.hitType = 2;
            hit.normal = (float3)(0.0f, 1.0f, 0.0f);
            hit.color = (float3)(0.28f, 0.30f, 0.34f);
            hit.emissive = 0.02f;
        }
    }

    // Saturn ring, aligned to the planet.
    float saturnInner = 1.45f;
    float saturnOuter = 2.25f;
    float tRing = intersect_ring_xz(ro, rd, spheres[6].center, saturnInner, saturnOuter);
    if (tRing > 0.0f && tRing < hit.t) {
        float3 p = ro + rd * tRing;
        float radius = length((float2)(p.x - spheres[6].center.x, p.z - spheres[6].center.z));
        float mask = ring_mask(radius, saturnInner, saturnOuter);
        hit.t = tRing;
        hit.hitType = 2;
        hit.normal = (float3)(0.0f, 1.0f, 0.0f);
        hit.color = ((float3)(0.85f, 0.78f, 0.62f) * mask) + (float3)(0.10f, 0.08f, 0.05f) * (1.0f - mask);
        hit.emissive = 0.01f;
    }

    return hit;
}

float3 background_color(float3 rd) {
    float blend = clamp(0.5f * (rd.y + 1.0f), 0.0f, 1.0f);
    float3 bgTop = (float3)(0.00f, 0.00f, 0.03f);
    float3 bgBottom = (float3)(0.0f, 0.0f, 0.0f);
    float3 background = bgBottom * (1.0f - blend) + bgTop * blend;

    // Dense small stars
    float s1 = star_noise(rd.xz * 220.0f);
    float s2 = star_noise((rd.xy + (float2)(0.37f, 0.11f)) * 410.0f);
    float s3 = star_noise((float2)(rd.z, rd.x) + (float2)(0.73f, 0.29f)) * 760.0f;

    if (s1 > 0.9978f) background += (float3)(0.65f, 0.70f, 0.78f) * (s1 - 0.9978f) * 260.0f;
    if (s2 > 0.9987f) background += (float3)(0.85f, 0.84f, 0.78f) * (s2 - 0.9987f) * 650.0f;
    if (s3 > 0.99925f) background += (float3)(0.70f, 0.75f, 0.95f) * (s3 - 0.99925f) * 1300.0f;

    return clamp(background, 0.0f, 1.0f);
}

float3 shade(float3 ro, float3 rd, __global const Sphere* spheres, int sphereCount) {
    float3 background = background_color(rd);
    HitInfo hit = trace_scene(ro, rd, spheres, sphereCount);

    if (hit.hitType == 0) {
        return background;
    }

    float3 p = ro + rd * hit.t;
    float3 lightDir = normalize(spheres[0].center - p);
    float diff = fmax(dot(hit.normal, lightDir), 0.0f);
    float distToSun = length(spheres[0].center - p);
    float attenuation = 1.0f / (1.0f + 0.03f * distToSun * distToSun);

    float3 ambient = 0.08f * hit.color;
    float3 direct = diff * attenuation * hit.color * 4.0f;
    float3 emissive = hit.color * hit.emissive;

    // Add a little rim lighting for rings/orbits so they remain visible.
    float rim = pow(clamp(1.0f - fmax(dot(hit.normal, -rd), 0.0f), 0.0f, 1.0f), 2.0f);
    float3 rimLight = hit.color * rim * (hit.hitType == 2 ? 0.8f : 0.18f);

    return clamp(ambient + direct + emissive + rimLight + background * 0.03f, 0.0f, 1.0f);
}

__kernel void render(
    __global uchar4* pixels,
    __global const Sphere* spheres,
    const int sphereCount,
    const int width,
    const int height,
    const float time,
    const float yaw,
    const float pitch,
    const float camDistance
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

    float cp = cos(pitch);
    float sp = sin(pitch);
    float cy = cos(yaw);
    float sy = sin(yaw);

    float3 camPos = (float3)(
        camDistance * cp * sy,
        camDistance * sp,
        camDistance * cp * cy
    );

    float3 target = (float3)(0.0f, 0.0f, 0.0f);
    float3 forward = normalize(target - camPos);
    float3 right = normalize(cross((float3)(0.0f, 1.0f, 0.0f), forward));
    float3 up = cross(forward, right);

    float fov = 1.15f;
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
    std::string log(logSize, char(0));
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
        SDL_PIXELFORMAT_ABGR8888,
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

    float yaw = 0.0f;
    float pitch = 0.28f;
    float camDistance = 34.0f;
    bool dragging = false;
    int lastMouseX = 0;
    int lastMouseY = 0;

    while (running) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = false;
            }
            if (event.type == SDL_KEYDOWN) {
                if (event.key.keysym.sym == SDLK_ESCAPE) running = false;
                if (event.key.keysym.sym == SDLK_LEFT) yaw -= 0.08f;
                if (event.key.keysym.sym == SDLK_RIGHT) yaw += 0.08f;
                if (event.key.keysym.sym == SDLK_UP) pitch += 0.06f;
                if (event.key.keysym.sym == SDLK_DOWN) pitch -= 0.06f;
                if (event.key.keysym.sym == SDLK_w) camDistance -= 1.2f;
                if (event.key.keysym.sym == SDLK_s) camDistance += 1.2f;
            }
            if (event.type == SDL_MOUSEBUTTONDOWN && event.button.button == SDL_BUTTON_LEFT) {
                dragging = true;
                lastMouseX = event.button.x;
                lastMouseY = event.button.y;
            }
            if (event.type == SDL_MOUSEBUTTONUP && event.button.button == SDL_BUTTON_LEFT) {
                dragging = false;
            }
            if (event.type == SDL_MOUSEMOTION && dragging) {
                int dx = event.motion.x - lastMouseX;
                int dy = event.motion.y - lastMouseY;
                yaw += dx * 0.005f;
                pitch -= dy * 0.005f;
                lastMouseX = event.motion.x;
                lastMouseY = event.motion.y;
            }
            if (event.type == SDL_MOUSEWHEEL) {
                camDistance -= event.wheel.y * 1.2f;
            }
        }

        if (pitch > 1.35f) pitch = 1.35f;
        if (pitch < -1.35f) pitch = -1.35f;
        if (camDistance < 8.0f) camDistance = 8.0f;
        if (camDistance > 90.0f) camDistance = 90.0f;

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

        
        spheres[1] = makeSphere(0, 0, 0, 0.35f, 0.60f, 0.58f, 0.56f); // Mercury
        spheres[1].center = orbitPos(5.0f, 1.8f, 0.0f);

        spheres[2] = makeSphere(0, 0, 0, 0.50f, 0.88f, 0.74f, 0.50f); // Venus
        spheres[2].center = orbitPos(7.2f, 1.3f, 1.1f);

        spheres[3] = makeSphere(0, 0, 0, 0.55f, 0.22f, 0.45f, 0.90f); // Earth
        spheres[3].center = orbitPos(10.0f, 1.0f, 2.0f);

        spheres[4] = makeSphere(0, 0, 0, 0.42f, 0.78f, 0.36f, 0.22f); // Mars
        spheres[4].center = orbitPos(13.0f, 0.8f, 0.3f);

        spheres[5] = makeSphere(0, 0, 0, 1.20f, 0.82f, 0.70f, 0.52f); // Jupiter
        spheres[5].center = orbitPos(17.0f, 0.55f, 1.8f);

        spheres[6] = makeSphere(0, 0, 0, 1.00f, 0.90f, 0.82f, 0.62f); // Saturn
        spheres[6].center = orbitPos(21.0f, 0.42f, 2.6f);

        spheres[7] = makeSphere(0, 0, 0, 0.78f, 0.56f, 0.80f, 0.86f); // Uranus
        spheres[7].center = orbitPos(25.0f, 0.32f, 0.7f);

        spheres[8] = makeSphere(0, 0, 0, 0.76f, 0.20f, 0.34f, 0.78f); // Neptune
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
        checkErr(clSetKernelArg(kernel, 6, sizeof(float), &yaw), "clSetKernelArg 6");
        checkErr(clSetKernelArg(kernel, 7, sizeof(float), &pitch), "clSetKernelArg 7");
        checkErr(clSetKernelArg(kernel, 8, sizeof(float), &camDistance), "clSetKernelArg 8");

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
