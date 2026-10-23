#include "glad.h"
#include <GLFW/glfw3.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cuda_gl_interop.h>
#include <random>
#include <string_view>
#include <utility>
#include <vector>

#define CUDA_CHECK(call)                            \
    do {                                            \
        cudaError_t err = (call);                   \
        if (err != cudaSuccess) {                   \
            std::fprintf(stderr,                    \
                         "CUDA error: %s:%d: %s\n", \
                         __FILE__,                  \
                         __LINE__,                  \
                         cudaGetErrorString(err));  \
            std::exit(1);                           \
        }                                           \
    } while (0)

const char* vertexShaderSource   = ("#version 400 core\n"
                                    "layout(location = 0) in vec2 position;\n"
                                    "layout(location = 1) in vec2 texCoord;\n"
                                    "out vec2 TexCoord;\n"
                                    "void main()\n"
                                    "{\n"
                                    "    gl_Position = vec4(position, 0.0, 1.0);\n"
                                    "    TexCoord = texCoord;\n"
                                    "}\n");
const char* fragmentShaderSource = ("#version 400 core\n"
                                    "in vec2 TexCoord;\n"
                                    "out vec4 FragColor;\n"
                                    "uniform float zoom;\n"
                                    "uniform sampler2D gridTexture;\n"
                                    "void main()\n"
                                    "{\n"
                                    "vec2 uv = TexCoord;\n"
                                    "uv = (uv - 0.5) / zoom + 0.5;\n"
                                    "float cell = texture(gridTexture, uv).r;\n"
                                    "FragColor = vec4(cell, cell, cell, 1.0);\n"
                                    "}\n");

GLFWwindow* opengl_init(int screen_width, int screen_height) {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* window = glfwCreateWindow(screen_width, screen_height, "LearnOpenGL", NULL, NULL);
    if (window == NULL) {
        std::fprintf(stderr, "Failed to create window\n");
        glfwTerminate();
        return NULL;
    }
    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
        std::fprintf(stderr, "Failed to initialize GLAD\n");
        return NULL;
    }
    glViewport(0, 0, screen_width, screen_height);
    return window;
}

struct GLGrid {
    GLuint vbo;
    GLuint vao;
    GLuint ebo;
    GLuint texture;

    size_t n_vertices;
    size_t n_indices;

    GLGrid(std::uint32_t width, std::uint32_t height) {
        // clang-format off
        // Texture coordinate:
        // (0, 0) -- (1, 0)
        // (0, 1) -- (1, 1)
        // OpenGl coordinate
        // (-1, -1) -- (1, -1)
        // (-1, 1) -- (1, 1)
        float vertices[] = {
            // position     // texture coordinate
            -1.0f, -1.0f,   0.0f, 0.0f,
            1.0f, -1.0f,    1.0f, 0.0f,
            1.0f,  1.0f,    1.0f, 1.0f,
            -1.0f,  1.0f,   0.0f, 1.0f
        };
        unsigned int indices[] = {
            0, 1, 2,
            2, 3, 0
        };
        // clang-format on

        // Array Buffer (Where is the data)
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        // Vertex arrays (How to interpret the data)
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        // Position
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*) 0);
        glEnableVertexAttribArray(0);
        // Textuer coordinate
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*) (2 * sizeof(float)));
        glEnableVertexAttribArray(1);
        // Element array buffer (How to use the data)
        glGenBuffers(1, &ebo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, width, height, 0, GL_RED, GL_UNSIGNED_BYTE, nullptr);
    }
};

struct dim2 {
    std::uint32_t x;
    std::uint32_t y;
};
__device__ dim2 fixdim2(const dim2& golGridDim, std::int64_t x, std::int64_t y) {
    dim2 fixeddim2;

    if (x < 0)
        fixeddim2.x = golGridDim.x - 1;
    else if (x >= golGridDim.x)
        fixeddim2.x = 0;
    else
        fixeddim2.x = x;
    if (y < 0)
        fixeddim2.y = golGridDim.y - 1;
    else if (y >= golGridDim.y)
        fixeddim2.y = 0;
    else
        fixeddim2.y = y;

    return fixeddim2;
}
__global__ void golGenerate(uint8_t* golGridNext) {
}
__global__ void randomizeGrid(std::uint8_t* grid, std::uint32_t width, std::uint32_t height, std::uint32_t seed) {
    std::uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;
    std::uint32_t index = y * width + x;

    // Source: https://nullprogram.com/blog/2018/07/31/
    uint32_t h = index + seed;
    h ^= h >> 16;
    h *= 0x45d9f3bU;
    h ^= h >> 16;
    h *= 0x45d9f3b;
    h ^= h >> 16;

    grid[index] = (h % 100 < 25) ? 255 : 0; // 25 percent chance to be alive
}
__global__ void golNextGeneration(const std::uint8_t* golGridPrev, std::uint8_t* golGridNext, dim2 golGridDim, cudaSurfaceObject_t output) {
    std::int64_t xTarget;
    std::int64_t yTarget;

    xTarget = blockIdx.x * blockDim.x + threadIdx.x;
    yTarget = blockIdx.y * blockDim.y + threadIdx.y;
    if ((std::uint32_t) xTarget >= golGridDim.x || (std::uint32_t) yTarget >= golGridDim.y)
        return;

    std::uint8_t n = 0;
    for (std::int64_t x = xTarget - 1; x <= xTarget + 1; x++) {
        for (std::int64_t y = yTarget - 1; y <= yTarget + 1; y++) {
            if (x == xTarget && y == yTarget)
                continue;
            const dim2 fixeddim2 = fixdim2(golGridDim, x, y);
            n += golGridPrev[fixeddim2.y * golGridDim.x + fixeddim2.x] == 255;
        }
    }
    std::uint32_t index = yTarget * golGridDim.x + xTarget;
    if (n == 3)
        golGridNext[index] = 255;
    else if (n == 2)
        golGridNext[index] = golGridPrev[index];
    else
        golGridNext[index] = 0;

    surf2Dwrite(golGridNext[index], output, xTarget * sizeof(uint8_t), yTarget);
}
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    float* zoom = static_cast<float*>(glfwGetWindowUserPointer(window));
    if (yoffset > 0)
        *zoom *= 1.1f;
    else
        *zoom /= 1.1f;
    *zoom = std::clamp(*zoom, 0.1f, 1000.0f);
}

int main() {
    constexpr std::uint32_t screenWidth  = 1920;
    constexpr std::uint32_t screenHeight = 1080;
    constexpr std::uint32_t factor       = 17;
    constexpr std::uint32_t width        = screenWidth * factor;
    constexpr std::uint32_t height       = screenHeight * factor;
    constexpr std::uint32_t blockWidth   = 16;
    constexpr std::uint32_t blockHeight  = 16;

    dim3 block(blockWidth, blockHeight);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    uint8_t* golGridGPUPrev;
    uint8_t* golGridGPUNext;
    CUDA_CHECK(cudaMalloc(&golGridGPUPrev, width * height));
    CUDA_CHECK(cudaMalloc(&golGridGPUNext, width * height));
    randomizeGrid<<<grid, block>>>(golGridGPUPrev, width, height, time(nullptr));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    GLFWwindow* window = opengl_init(1920, 1080);
    GLint maxTextureSize;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxTextureSize);
    std::printf("GL_MAX_TEXTURE_SIZE = %d\n", maxTextureSize);
    GLGrid glGrid(width, height);
    // Create cuda resource
    cudaGraphicsResource* cudaResource;
    // Register it (so CUDA knows about it)
    cudaGraphicsGLRegisterImage(&cudaResource, glGrid.texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);

    // Create the shader
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    glUseProgram(shaderProgram);

    // Attach the texture to the shader
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, glGrid.texture);
    GLint location = glGetUniformLocation(shaderProgram, "gridTexture");
    glUniform1i(location, 0);
    GLint zoomLocation = glGetUniformLocation(shaderProgram, "zoom");
    float zoom         = (double) factor;
    glUniform1f(zoomLocation, zoom);
    glfwSetWindowUserPointer(window, &zoom);
    glfwSetScrollCallback(window, scroll_callback);

    while (!glfwWindowShouldClose(window)) {
        /* CUDA <-> OpenGL */
        // Map it (so CUDA can use it)
        cudaGraphicsMapResources(1, &cudaResource);
        // Get the array from the CUDA resource to be used inside the CUDA kernel
        cudaArray_t array;
        cudaGraphicsSubResourceGetMappedArray(&array, cudaResource, 0, 0);
        // Create a view of the array (there are so much steps...)
        cudaResourceDesc resourceDesc{};
        resourceDesc.resType         = cudaResourceTypeArray;
        resourceDesc.res.array.array = array;
        cudaSurfaceObject_t surface;
        cudaCreateSurfaceObject(&surface, &resourceDesc);

        // FINALLY we can use it in our kernel
        golNextGeneration<<<grid, block>>>(golGridGPUPrev, golGridGPUNext, {width, height}, surface);
        cudaDeviceSynchronize();

        // Destroy the surface
        cudaDestroySurfaceObject(surface);
        // Unmap the resource
        cudaGraphicsUnmapResources(1, &cudaResource);
        // Swap prev and next
        std::swap(golGridGPUPrev, golGridGPUNext);

        // OpenGL Drawing
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(shaderProgram);
        glUniform1f(zoomLocation, zoom);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, glGrid.texture);
        glBindVertexArray(glGrid.vao);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }


    return 0;
}
