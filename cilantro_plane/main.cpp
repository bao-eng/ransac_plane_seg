#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <cilantro/utilities/point_cloud.hpp>
#include <cilantro/model_estimation/ransac_hyperplane_estimator.hpp>
#include <cilantro/visualization.hpp>

using namespace std;
using namespace Eigen;

void callback(bool &re_estimate)
{
    re_estimate = true;
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cout << "Please provide path to input file." << std::endl;
        return 0;
    }

    ifstream input(argv[1]);

    float max_threshold = 0;
    size_t num_points = 0;
    input >> max_threshold >> num_points;

    cilantro::PointCloud3f cloud;
    cloud.points.conservativeResize(NoChange, cloud.points.cols() + num_points);

    float x, y, z;
    for (auto i = 0; i < num_points; i++)
    {
        input >> x >> y >> z;
        cloud.points.col(i) = Vector3f(x, y, z);
    }

    cilantro::Visualizer viz("HyperplaneRANSACEstimator example", "disp");
    viz.setCameraPose(60, 60, 60, 0, 0, 0, 200, 200, 201);
    viz.setPerspectiveProjectionMatrix(640, 480, 525, 525, 319.5, 239.5, 0.1, 1000); //default values from visualizer.cpp with increazed zFar to avoid far plane clipping

    bool re_estimate = false;
    viz.registerKeyboardCallback('a', std::bind(callback, std::ref(re_estimate)));

    std::cout << "Press 'a' for a new estimate" << std::endl;

    cilantro::PointCloud3f planar_cloud;

    viz.addObject<cilantro::PointCloudRenderable>("cloud", cloud, cilantro::RenderingProperties().setUseLighting(1));

    while (!viz.wasStopped())
    {
        if (re_estimate)
        {
            re_estimate = false;

            cilantro::PlaneRANSACEstimator3f<> pe(cloud.points);
            pe.setMaxInlierResidual(max_threshold).setTargetInlierCount((size_t)(0.5 * cloud.size())).setMaxNumberOfIterations(250).setReEstimationStep(true);

            Eigen::Hyperplane<float, 3> plane = pe.estimate().getModel();
            const auto &inliers = pe.getModelInliers();

            std::cout << "RANSAC iterations: " << pe.getNumberOfPerformedIterations() << ", inlier count: " << pe.getNumberOfInliers() << std::endl;

            planar_cloud = cilantro::PointCloud3f(cloud, inliers);
            viz.addObject<cilantro::PointCloudRenderable>("plane", planar_cloud.points, cilantro::RenderingProperties().setPointColor(1, 0, 0).setPointSize(3.0));

            std::cout << "Press 'a' for a new estimate" << std::endl;
        }
        viz.spinOnce();
    }
    return 0;
}