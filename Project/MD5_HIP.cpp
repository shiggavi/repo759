#include <hip/hip_runtime.h>
#include <iostream>
#include <cstring>
#include <cmath>
#include <chrono>

// Constants for MD5
__constant__ uint32_t K[64] = {
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
    0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
    0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
    0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
    0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
    0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391
};

__constant__ uint32_t S[64] = {
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
    5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21
};

// Rotate left function for HIP
__device__ uint32_t rotateLeft(uint32_t x, uint32_t n) {
    return (x << n) | (x >> (32 - n));
}

// MD5 context structure
struct MD5Context {
    uint64_t size;
    uint32_t buffer[4];
    uint8_t input[64];
    uint8_t digest[16];
};

// Device function to initialize MD5 context
__device__ void md5Init(MD5Context &ctx) {
    ctx.size = 0;
    ctx.buffer[0] = 0x67452301;
    ctx.buffer[1] = 0xefcdab89;
    ctx.buffer[2] = 0x98badcfe;
    ctx.buffer[3] = 0x10325476;
}

// Device function to process MD5 steps
__device__ void md5Step(uint32_t *buffer, const uint32_t *input) {
    uint32_t AA = buffer[0];
    uint32_t BB = buffer[1];
    uint32_t CC = buffer[2];
    uint32_t DD = buffer[3];

    uint32_t E;
    unsigned int j;

    for (unsigned int i = 0; i < 64; ++i) {
        switch (i / 16) {
            case 0:
                E = (BB & CC) | (~BB & DD);
                j = i;
                break;
            case 1:
                E = (BB & DD) | (CC & ~DD);
                j = ((i * 5) + 1) % 16;
                break;
            case 2:
                E = BB ^ CC ^ DD;
                j = ((i * 3) + 5) % 16;
                break;
            default:
                E = CC ^ (BB | ~DD);
                j = (i * 7) % 16;
                break;
        }

        uint32_t temp = DD;
        DD = CC;
        CC = BB;
        BB = BB + rotateLeft(AA + E + K[i] + input[j], S[i]);
        AA = temp;
    }

    buffer[0] += AA;
    buffer[1] += BB;
    buffer[2] += CC;
    buffer[3] += DD;
}

// Device function to update MD5 context with new input
__device__ void md5Update(MD5Context &ctx, const uint8_t *input_buffer, size_t input_len) {
    uint32_t input[16];
    unsigned int offset = static_cast<unsigned int>(ctx.size % 64);
    ctx.size += static_cast<uint64_t>(input_len);

    for (size_t i = 0; i < input_len; ++i) {
        ctx.input[offset++] = input_buffer[i];

        if (offset == 64) {
            for (size_t j = 0; j < 16; ++j) {
                input[j] = static_cast<uint32_t>(ctx.input[(j * 4) + 3]) << 24 |
                           static_cast<uint32_t>(ctx.input[(j * 4) + 2]) << 16 |
                           static_cast<uint32_t>(ctx.input[(j * 4) + 1]) << 8 |
                           static_cast<uint32_t>(ctx.input[(j * 4)]);
            }
            md5Step(ctx.buffer, input);
            offset = 0;
        }
    }
}

// Device function to finalize MD5 computation
__device__ void md5Finalize(MD5Context &ctx) {
    uint32_t input[16];
    unsigned int offset = static_cast<unsigned int>(ctx.size % 64);
    unsigned int padding_length = (offset < 56) ? (56 - offset) : (120 - offset);

    const uint8_t PADDING[64] = {0x80};
    md5Update(ctx, PADDING, padding_length);
    ctx.size -= static_cast<uint64_t>(padding_length);

    for (unsigned int j = 0; j < 14; ++j) {
        input[j] = static_cast<uint32_t>(ctx.input[(j * 4) + 3]) << 24 |
                   static_cast<uint32_t>(ctx.input[(j * 4) + 2]) << 16 |
                   static_cast<uint32_t>(ctx.input[(j * 4) + 1]) << 8 |
                   static_cast<uint32_t>(ctx.input[(j * 4)]);
    }
    input[14] = static_cast<uint32_t>(ctx.size * 8);
    input[15] = static_cast<uint32_t>(ctx.size >> 32);

    md5Step(ctx.buffer, input);

    for (unsigned int i = 0; i < 4; ++i) {
        ctx.digest[(i * 4) + 0] = static_cast<uint8_t>(ctx.buffer[i] & 0x000000FF);
        ctx.digest[(i * 4) + 1] = static_cast<uint8_t>((ctx.buffer[i] & 0x0000FF00) >> 8);
        ctx.digest[(i * 4) + 2] = static_cast<uint8_t>((ctx.buffer[i] & 0x00FF0000) >> 16);
        ctx.digest[(i * 4) + 3] = static_cast<uint8_t>((ctx.buffer[i] & 0xFF000000) >> 24);
    }
}

// HIP kernel for MD5 brute-force password cracking
__global__ void md5String(const char *charset, int charsetSize, int passwordLength,
                         const char *targetHash, char *result, int *found, uint64_t totalCombinations) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = gridDim.x * blockDim.x;

    for (uint64_t i = idx; i < totalCombinations; i += stride) {
        if (atomicAdd(found, 0)) return;

        char password[8] = {0};
        char digest[33] = {0};
        uint64_t temp = i;

        // Generate the password
        for (int j = 0; j < passwordLength; ++j) {
            password[j] = charset[temp % charsetSize];
            temp /= charsetSize;
        }

        // Compute the MD5 hash for the password
        MD5Context ctx;
        md5Init(ctx);
        md5Update(ctx, reinterpret_cast<const uint8_t *>(password), passwordLength);
        md5Finalize(ctx);

        // Convert the digest to a hex string
        for (int j = 0; j < 16; ++j) {
            sprintf(&digest[j * 2], "%02x", ctx.digest[j]);
        }

        // Compare the generated hash with the target hash
        if (strcmp(digest, targetHash) == 0) {
            atomicExch(found, 1);
            strcpy(result, password);
            return;
        }
    }
}

int main(int argc, char *argv[]) {
    const char *targetHash = "b9224147aa182999e7975c828de15d2f";
    const char charset[] = "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    const int passwordLength = 7;
    const int charsetSize = strlen(charset);
    uint64_t totalCombinations = pow(charsetSize, passwordLength);

    const int threadsPerBlock = 1024;
    const int maxBlocksPerGrid = 2147483647;
    const int blocksPerGrid = min((totalCombinations + threadsPerBlock - 1) / threadsPerBlock, maxBlocksPerGrid);

    char *d_charset, *d_targetHash, *d_result;
    int *d_found;

    hipMalloc(&d_charset, sizeof(char) * charsetSize);
    hipMalloc(&d_targetHash, sizeof(char) * 33);
    hipMalloc(&d_result, sizeof(char) * 8);
    hipMalloc(&d_found, sizeof(int));

    hipMemcpy(d_charset, charset, sizeof(char) * charsetSize, hipMemcpyHostToDevice);
    hipMemcpy(d_targetHash, targetHash, sizeof(char) * 33, hipMemcpyHostToDevice);
    hipMemset(d_result, 0, sizeof(char) * 8);
    hipMemset(d_found, 0, sizeof(int));

    auto start = std::chrono::high_resolution_clock::now();

    hipLaunchKernelGGL(md5String, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0,
                       d_charset, charsetSize, passwordLength, d_targetHash, d_result, d_found, totalCombinations);
    hipDeviceSynchronize();

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);

    char result[8] = {0};
    int found = 0;
    hipMemcpy(result, d_result, sizeof(char) * 8, hipMemcpyDeviceToHost);
    hipMemcpy(&found, d_found, sizeof(int), hipMemcpyDeviceToHost);

    hipFree(d_charset);
    hipFree(d_targetHash);
    hipFree(d_result);
    hipFree(d_found);

    if (found) {
        std::cout << "Password Cracked: " << result << std::endl;
        std::cout << "Password cracked in: " << duration.count() << " seconds" << std::endl;
    } else {
        std::cout << "Password not found." << std::endl;
    }

    return 0;
}
