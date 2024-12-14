#include <omp.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include <iomanip>
#include <ctime>
#include <chrono>
#include <atomic>
#include "MD5_CPU.h"


// Rotates a 32-bit word left by n bits
uint32_t rotateLeft(uint32_t x, uint32_t n)
{
    return (x << n) | (x >> (32 - n));
}

// Step on 512 bits of input with the main MD5 algorithm.
void md5Step(uint32_t *buffer, uint32_t *input)
{
    uint32_t AA = buffer[0];
    uint32_t BB = buffer[1];
    uint32_t CC = buffer[2];
    uint32_t DD = buffer[3];

    uint32_t E;

    unsigned int j;

    for (unsigned int i = 0; i < 64; ++i)
    {
        switch (i / 16)
        {
        case 0:
            E = F(BB, CC, DD);
            j = i;
            break;
        case 1:
            E = G(BB, CC, DD);
            j = ((i * 5) + 1) % 16;
            break;
        case 2:
            E = H(BB, CC, DD);
            j = ((i * 3) + 5) % 16;
            break;
        default:
            E = I(BB, CC, DD);
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

// Initialize a context
void md5Init(MD5Context &ctx)
{
    ctx.size = 0;
    ctx.buffer[0] = A;
    ctx.buffer[1] = B;
    ctx.buffer[2] = C;
    ctx.buffer[3] = D;
}

// Add some amount of input to the context
void md5Update(MD5Context &ctx, const uint8_t *input_buffer, size_t input_len)
{
    uint32_t input[16];
    unsigned int offset = static_cast<unsigned int>(ctx.size % 64);
    ctx.size += static_cast<uint64_t>(input_len);

    for (size_t i = 0; i < input_len; ++i)
    {
        ctx.input[offset++] = input_buffer[i];
        if (offset % 64 == 0)
        {
            for (size_t j = 0; j < 16; ++j)
            {
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

// Pad the current input and finalize the MD5 hash
void md5Finalize(MD5Context &ctx)
{
    uint32_t input[16];
    unsigned int offset = static_cast<unsigned int>(ctx.size % 64);
    unsigned int padding_length = (offset < 56) ? (56 - offset) : (120 - offset);

    md5Update(ctx, PADDING, padding_length);
    ctx.size -= static_cast<uint64_t>(padding_length);

    for (unsigned int j = 0; j < 14; ++j)
    {
        input[j] = static_cast<uint32_t>(ctx.input[(j * 4) + 3]) << 24 |
                   static_cast<uint32_t>(ctx.input[(j * 4) + 2]) << 16 |
                   static_cast<uint32_t>(ctx.input[(j * 4) + 1]) << 8 |
                   static_cast<uint32_t>(ctx.input[(j * 4)]);
    }
    input[14] = static_cast<uint32_t>(ctx.size * 8);
    input[15] = static_cast<uint32_t>(ctx.size >> 32);

    md5Step(ctx.buffer, input);

    for (unsigned int i = 0; i < 4; ++i)
    {
        ctx.digest[(i * 4) + 0] = static_cast<uint8_t>(ctx.buffer[i] & 0x000000FF);
        ctx.digest[(i * 4) + 1] = static_cast<uint8_t>((ctx.buffer[i] & 0x0000FF00) >> 8);
        ctx.digest[(i * 4) + 2] = static_cast<uint8_t>((ctx.buffer[i] & 0x00FF0000) >> 16);
        ctx.digest[(i * 4) + 3] = static_cast<uint8_t>((ctx.buffer[i] & 0xFF000000) >> 24);
    }
}

// Run the MD5 algorithm on the provided input and store the digest in result
void md5String(const std::string &input, char *md5_result)
{
    MD5Context ctx;
    md5Init(ctx);
    md5Update(ctx, reinterpret_cast<const uint8_t *>(input.c_str()), input.length());
    md5Finalize(ctx);
    for (int i = 0; i < 16; ++i)
    {
        sprintf(&md5_result[i * 2], "%02x", ctx.digest[i]);
    }
}

int main(int argc, char *argv[])
{
    const char *targetHash = "b9224147aa182999e7975c828de15d2f\0"; // MD5 hash of '8i9G2aa'
    const char charset[] = "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    const int passwordLength = 7;

    int num_threads = std::stod(argv[1]); // Number of threads
    omp_set_num_threads(num_threads);

    char digest[32];
    char password[8] = {0}; // Null-terminated string to hold the password
    int charsetSize = strlen(charset);
    uint64_t totalCombinations = 1;
    uint64_t temp = 0;
    std::atomic<bool> found(false);
    double start = omp_get_wtime();

    for (int i = 0; i < passwordLength; i++)
        totalCombinations *= charsetSize;
    std::cout << "Total Combinations: " << totalCombinations << std::endl;

    #pragma omp parallel for private(temp, password, digest) shared(found) schedule(dynamic)
    for (uint64_t iteration = 0; iteration < totalCombinations; ++iteration)
    {
        if (found.load()) exit(0);

        temp = iteration;
        for (int i = 0; i < passwordLength; ++i)
        {
            password[i] = charset[temp % charsetSize];
            temp /= charsetSize;
        }

        md5String(password, digest);
       
        if (strcmp(digest, targetHash) == 0)
        {
            std::cout << "Password Cracked - " << password << std::endl;
            std::cout << "time: " << (omp_get_wtime() - start) << "s\n";
            found.store(true);
        }
    }

    if (!found.load())
    {
        std::cout << "Password not found." << std::endl;
    }
    return 0;
}
