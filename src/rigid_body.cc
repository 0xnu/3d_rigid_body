// src/rigid_body.cc

#include <cmath>
#include <Eigen/Dense>
#include "rigid_body.h"

RigidBody::RigidBody(double m, const Eigen::Vector3d& i, const Eigen::Vector3d& v, 
                     const Eigen::Vector3d& p, const Eigen::Vector3d& av, 
                     const Eigen::Quaterniond& o, double f, double r, double d, double g)
    : mass(m), inertia(i), velocity(v), position(p), angularVelocity(av), orientation(o),
      friction(f), restitution(r), damping(d), gravity(g) {}

void RigidBody::applyForce(const Eigen::Vector3d& force, double duration) {
    Eigen::Vector3d acceleration = force / mass;
    velocity += acceleration * duration;
    position += velocity * duration;
}

void RigidBody::applyTorque(const Eigen::Vector3d& torque, double duration) {
    Eigen::Vector3d angularAcceleration = inertia.cwiseInverse().cwiseProduct(torque);
    angularVelocity += angularAcceleration * duration;
    
    // Update orientation using quaternion integration
    Eigen::Quaterniond angularVelocityQuat(0, angularVelocity.x(), angularVelocity.y(), angularVelocity.z());
    Eigen::Quaterniond orientationDelta = Eigen::Quaterniond::Identity().slerp(duration / 2.0, angularVelocityQuat * orientation);
    orientation = (orientation * orientationDelta).normalized();
}

void RigidBody::applyFluidFriction(double fluidDensity, double dragCoefficient) {
    double speed = velocity.norm();
    Eigen::Vector3d dragForce = -0.5 * fluidDensity * speed * speed * dragCoefficient * velocity.normalized();
    applyForce(dragForce, 1.0); // Assuming a time step of 1 second
}

void RigidBody::applyGravity() {
    Eigen::Vector3d gravitationalForce(0, 0, -mass * gravity);
    applyForce(gravitationalForce, 1.0); // Assuming a time step of 1 second
}

void RigidBody::applyDamping() {
    velocity *= damping;
    angularVelocity *= damping;
}

void RigidBody::handleElasticCollision(RigidBody& other) {
    Eigen::Vector3d normal = (other.position - position).normalized();
    Eigen::Vector3d relativeVelocity = other.velocity - velocity;

    double impulse = -(1 + restitution) * relativeVelocity.dot(normal) /
                     (1 / mass + 1 / other.mass);

    Eigen::Vector3d impulseVector = impulse * normal;

    velocity -= impulseVector / mass;
    other.velocity += impulseVector / other.mass;

    // Angular impulse
    Eigen::Vector3d r1 = position - position; // Relative position of collision point
    Eigen::Vector3d r2 = other.position - other.position;

    Eigen::Vector3d angularImpulse1 = r1.cross(impulseVector);
    Eigen::Vector3d angularImpulse2 = r2.cross(-impulseVector);

    angularVelocity += inertia.cwiseInverse().cwiseProduct(angularImpulse1);
    other.angularVelocity += other.inertia.cwiseInverse().cwiseProduct(angularImpulse2);
}

Eigen::VectorXd RigidBody::getState() const {
    Eigen::VectorXd state(13);
    state << position, orientation.coeffs(), velocity, angularVelocity;
    return state;
}

void RigidBody::setState(const Eigen::VectorXd& state) {
    position = state.segment<3>(0);
    orientation.coeffs() = state.segment<4>(3);
    velocity = state.segment<3>(7);
    angularVelocity = state.segment<3>(10);
}