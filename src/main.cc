// src/main.cc

#include <iostream>
#include <vector>
#include <omp.h>
#include <Eigen/Dense>
#include "rigid_body.h"

int main() {
    std::vector<RigidBody> objects;

    // Create two rigid bodies with 3D properties
    objects.emplace_back(
        1.0,  // mass
        Eigen::Vector3d(0.1, 0.1, 0.1),  // inertia
        Eigen::Vector3d(10.0, 0.0, 0.0),  // velocity
        Eigen::Vector3d(0.0, 0.0, 0.0),  // position
        Eigen::Vector3d(5.0, 0.0, 0.0),  // angular velocity
        Eigen::Quaterniond::Identity(),  // orientation
        0.5,  // friction
        0.8,  // restitution
        0.99,  // damping
        9.81  // gravity
    );

    objects.emplace_back(
        2.0,  // mass
        Eigen::Vector3d(0.2, 0.2, 0.2),  // inertia
        Eigen::Vector3d(-5.0, 5.0, 0.0),  // velocity
        Eigen::Vector3d(10.0, 0.0, 0.0),  // position
        Eigen::Vector3d(-3.0, 1.0, 2.0),  // angular velocity
        Eigen::Quaterniond::Identity(),  // orientation
        0.4,  // friction
        0.9,  // restitution
        0.98,  // damping
        9.81  // gravity
    );

    double duration = 0.01;  // Simulation Time Step

    // Apply forces and update physics
    #pragma omp parallel for
    for (size_t i = 0; i < objects.size(); ++i) {
        objects[i].applyForce(Eigen::Vector3d(10.0, 0.0, 0.0), duration);
        objects[i].applyTorque(Eigen::Vector3d(5.0, 2.0, 1.0), duration);
        objects[i].applyFluidFriction(1.0, 0.05);
        objects[i].applyGravity();
        objects[i].applyDamping();
    }

    // Handle Collisions
    for (size_t i = 0; i < objects.size(); ++i) {
        for (size_t j = i + 1; j < objects.size(); ++j) {
            objects[i].handleElasticCollision(objects[j]);
        }
    }

    // Output Results
    for (const auto& obj : objects) {
        std::cout << "Position: " << obj.position.transpose()
                  << ", Velocity: " << obj.velocity.transpose()
                  << ", Orientation: " << obj.orientation.coeffs().transpose()
                  << ", Angular Velocity: " << obj.angularVelocity.transpose()
                  << ", Friction: " << obj.friction
                  << ", Restitution: " << obj.restitution
                  << ", Damping: " << obj.damping
                  << ", Gravity: " << obj.gravity << std::endl;

        // Output state vector
        Eigen::VectorXd state = obj.getState();
        std::cout << "State vector: " << state.transpose() << std::endl;
    }

    return 0;
}