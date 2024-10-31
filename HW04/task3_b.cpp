#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <chrono>
#include <omp.h>

// Constants
const double G = 1.0;           // Gravitational constant
const double softening = 0.1;   // Softening parameter to avoid singularities

struct Particle {
    double mass;
    double pos[3];
    double vel[3];
    double acc[3];
};

void computeAccelerations(std::vector<Particle>& particles) {
    size_t N = particles.size();

    #pragma omp parallel for
    for (size_t i = 0; i < N; ++i) {
        particles[i].acc[0] = 0.0;
        particles[i].acc[1] = 0.0;
        particles[i].acc[2] = 0.0;
    }

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            if (i != j) {
                double dx = particles[j].pos[0] - particles[i].pos[0];
                double dy = particles[j].pos[1] - particles[i].pos[1];
                double dz = particles[j].pos[2] - particles[i].pos[2];
                double inv_r3 = std::pow(dx * dx + dy * dy + dz * dz + softening * softening, -1.5);

                #pragma omp atomic
                particles[i].acc[0] += G * dx * inv_r3 * particles[j].mass;

                #pragma omp atomic
                particles[i].acc[1] += G * dy * inv_r3 * particles[j].mass;

                #pragma omp atomic
                particles[i].acc[2] += G * dz * inv_r3 * particles[j].mass;
            }
        }
    }
}

void leapfrogStep(std::vector<Particle>& particles, double dt) {
    #pragma omp parallel for
    for (auto& p : particles) {
        p.vel[0] += 0.5 * p.acc[0] * dt;
        p.vel[1] += 0.5 * p.acc[1] * dt;
        p.vel[2] += 0.5 * p.acc[2] * dt;
    }

    #pragma omp parallel for
    for (auto& p : particles) {
        p.pos[0] += p.vel[0] * dt;
        p.pos[1] += p.vel[1] * dt;
        p.pos[2] += p.vel[2] * dt;
    }

    computeAccelerations(particles);

    #pragma omp parallel for
    for (auto& p : particles) {
        p.vel[0] += 0.5 * p.acc[0] * dt;
        p.vel[1] += 0.5 * p.acc[1] * dt;
        p.vel[2] += 0.5 * p.acc[2] * dt;
    }
}

// Helper function to clear file contents at the start
void clearFile(const std::string& filename) {
    std::ofstream file(filename, std::ios::out | std::ios::trunc);  // Truncate file to overwrite
    if (file.is_open()) {
        file.close();
    } else {
        std::cerr << "Unable to clear file: " << filename << "\n";
    }
}

// Save timing results to a file
void saveTimingToFile(const std::string& filename, int threads, double time_ms) {
    std::ofstream file(filename, std::ios::app);  // Open in append mode for each entry
    if (file.is_open()) {
        file << threads << " " << time_ms << "\n";
        file.close();
    } else {
        std::cerr << "Unable to open file: " << filename << "\n";
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <number of particles> <simulation end time>\n";
        return 1;
    }

    size_t N = std::stoul(argv[1]);
    double tEnd = std::stod(argv[2]);
    double dt = 0.01;

    std::vector<Particle> particles(N);
    std::mt19937 rng(17);
    std::uniform_real_distribution<double> mass_dist(0.1, 1.0);
    std::normal_distribution<double> pos_vel_dist(0.0, 1.0);

    for (size_t i = 0; i < N; ++i) {
        particles[i].mass = mass_dist(rng);
        for (int j = 0; j < 3; ++j) {
            particles[i].pos[j] = pos_vel_dist(rng);
            particles[i].vel[j] = pos_vel_dist(rng);
            particles[i].acc[j] = 0.0;
        }
    }

    std::string filenames[3] = { "task4_static8.out", "task4_dynamic8.out", "task4_guided8.out" };
    std::string policies[3] = { "static", "dynamic", "guided" };

    // Clear files before starting the simulation
    for (const auto& filename : filenames) {
        clearFile(filename);
    }

    for (int p = 0; p < 3; ++p) {
        for (int threads = 1; threads <= 8; ++threads) {
            omp_set_num_threads(threads);

            if (policies[p] == "static") {
                omp_set_schedule(omp_sched_static, 0);
            } else if (policies[p] == "dynamic") {
                omp_set_schedule(omp_sched_dynamic, 0);
            } else if (policies[p] == "guided") {
                omp_set_schedule(omp_sched_guided, 0);
            }

            auto start = std::chrono::high_resolution_clock::now();
            double t = 0.0;

            while (t < tEnd) {
                leapfrogStep(particles, dt);
                t += dt;
            }

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapsed = end - start;

            saveTimingToFile(filenames[p], threads, elapsed.count());
        }
    }

    return 0;
}
