// src/rigid_body.h

#ifndef RIGIDBODY_H
#define RIGIDBODY_H

#include <Eigen/Dense>

class RigidBody {
public:
    double mass;
    Eigen::Vector3d inertia;
    Eigen::Vector3d velocity;
    Eigen::Vector3d position;
    Eigen::Vector3d angularVelocity;
    Eigen::Quaterniond orientation;
    double friction;
    double restitution;
    double damping;
    double gravity;

    RigidBody(double m, const Eigen::Vector3d& i, const Eigen::Vector3d& v, 
              const Eigen::Vector3d& p, const Eigen::Vector3d& av, 
              const Eigen::Quaterniond& o, double f, double r, double d, double g);

    void applyForce(const Eigen::Vector3d& force, double duration);
    void applyTorque(const Eigen::Vector3d& torque, double duration);
    void applyFluidFriction(double fluidDensity, double dragCoefficient);
    void applyGravity();
    void applyDamping();
    void handleElasticCollision(RigidBody& other);

    Eigen::VectorXd getState() const;
    void setState(const Eigen::VectorXd& state);
};

#endif // RIGIDBODY_H