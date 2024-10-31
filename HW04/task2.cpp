#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <chrono>

// Constants
const double G = 1.0;           // Gravitational constant
const double softening = 0.1;   // Softening parameter to avoid singularities

// Structure to hold particle data
struct Particle {
    double mass;
    double pos[3];
    double vel[3];
    double acc[3];
};

// Function to calculate accelerations using Newton's Law of Gravity
void computeAccelerations(std::vector<Particle>& particles) {
    size_t N = particles.size();

    // Reset accelerations to zero
    for (size_t i = 0; i < N; ++i) {
        particles[i].acc[0] = 0.0;
        particles[i].acc[1] = 0.0;
        particles[i].acc[2] = 0.0;
    }

    // Compute pairwise forces
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            if (i != j) {
                double dx = particles[j].pos[0] - particles[i].pos[0];
                double dy = particles[j].pos[1] - particles[i].pos[1];
                double dz = particles[j].pos[2] - particles[i].pos[2];
                double inv_r3 = std::pow(dx * dx + dy * dy + dz * dz + softening * softening, -1.5);

                particles[i].acc[0] += G * dx * inv_r3 * particles[j].mass;
                particles[i].acc[1] += G * dy * inv_r3 * particles[j].mass;
                particles[i].acc[2] += G * dz * inv_r3 * particles[j].mass;
            }
        }
    }
}

// Leapfrog integration to update particle positions and velocities
void leapfrogStep(std::vector<Particle>& particles, double dt) {
    // (1/2) Kick: Update velocities half-step
    for (auto& p : particles) {
        p.vel[0] += 0.5 * p.acc[0] * dt;
        p.vel[1] += 0.5 * p.acc[1] * dt;
        p.vel[2] += 0.5 * p.acc[2] * dt;
    }

    // Drift: Update positions
    for (auto& p : particles) {
        p.pos[0] += p.vel[0] * dt;
        p.pos[1] += p.vel[1] * dt;
        p.pos[2] += p.vel[2] * dt;
    }

    // Recompute accelerations
    computeAccelerations(particles);

    // (1/2) Kick: Complete velocity update
    for (auto& p : particles) {
        p.vel[0] += 0.5 * p.acc[0] * dt;
        p.vel[1] += 0.5 * p.acc[1] * dt;
        p.vel[2] += 0.5 * p.acc[2] * dt;
    }
}

// Save particle positions to CSV for visualization
void savePositionsToCSV(const std::vector<Particle>& particles, std::ofstream& file, int step) {
    file << step;  // Write the step number

    // Write particle positions as flat coordinates
    for (const auto& p : particles) {
        file << "," << p.pos[0] << "," << p.pos[1] << "," << p.pos[2];
    }
    file << "\n";  // New line after each step
}

int main(int argc, char* argv[]) {
    // Check for correct usage
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <number of particles> <simulation end time>\n";
        return 1;
    }

    // Parse command line arguments
    size_t N = std::stoul(argv[1]);  // Number of particles
    double tEnd = std::stod(argv[2]);  // Simulation end time
    double dt = 0.01;  // Time step

    // Initialize particles with random masses, positions, and velocities
    std::vector<Particle> particles(N);
    std::mt19937 rng(17);  // Random number generator (seeded for reproducibility)
    std::uniform_real_distribution<double> mass_dist(0.1, 1.0);
    std::normal_distribution<double> pos_vel_dist(0.0, 1.0);

    for (auto& p : particles) {
        p.mass = mass_dist(rng);
        for (int i = 0; i < 3; ++i) {
            p.pos[i] = pos_vel_dist(rng);
            p.vel[i] = pos_vel_dist(rng);
            p.acc[i] = 0.0;
        }
    }

    // Open CSV file for writing (optional)
    std::ofstream csvFile("positions.csv");

    // Measure simulation time
    auto start = std::chrono::high_resolution_clock::now();

    // Main simulation loop
    double t = 0.0;
    int step = 0;
    while (t < tEnd) {
        leapfrogStep(particles, dt);

        // (Optional) Save positions to CSV for visualization
        if (csvFile.is_open()) {
            savePositionsToCSV(particles, csvFile, step);
        }

        t += dt;
        ++step;
    }

    // Close the CSV file
    if (csvFile.is_open()) {
        csvFile.close();
    }

    // Measure and print the total simulation time
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Simulation completed in " << elapsed.count() << " seconds.\n";

    return 0;
}
