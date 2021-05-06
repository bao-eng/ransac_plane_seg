#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

struct Point3d {
  double X = 0.0;
  double Y = 0.0;
  double Z = 0.0;
};

using PointCloud = std::vector<Point3d>;

struct Plane {
  double A = 0.0;
  double B = 0.0;
  double C = 0.0;
  double D = 0.0;
};

bool CreatePlane(
    Plane& plane,
    const Point3d& p1,
    const Point3d& p2,
    const Point3d& p3) {
  const double matrix[3][3] =
    {{          0,           0,           0},  // this row is not used
     {p2.X - p1.X, p2.Y - p1.Y, p2.Z - p1.Z},
     {p3.X - p1.X, p3.Y - p1.Y, p3.Z - p1.Z}};

  auto getMatrixSignedMinor = [&matrix](size_t i, size_t j) {
    const auto& m = matrix;
    return (m[(i + 1) % 3][(j + 1) % 3] * m[(i + 2) % 3][(j + 2) % 3] -
            m[(i + 2) % 3][(j + 1) % 3] * m[(i + 1) % 3][(j + 2) % 3]);
  };

  const double A = getMatrixSignedMinor(0, 0);
  const double B = getMatrixSignedMinor(0, 1);
  const double C = getMatrixSignedMinor(0, 2);
  const double D = -(A * p1.X + B * p1.Y + C * p1.Z);

  const double norm_coeff = std::sqrt(A * A + B * B + C * C);

  const double kMinValidNormCoeff = 1.0e-6;
  if (norm_coeff < kMinValidNormCoeff) {
    return false;
  }

  plane.A = A / norm_coeff;
  plane.B = B / norm_coeff;
  plane.C = C / norm_coeff;
  plane.D = D / norm_coeff;

  return true;
}

double CalcDistance(const Plane& plane, const Point3d& point) {
  // assume that the plane is valid
  const auto numerator = std::abs(
      plane.A * point.X + plane.B * point.Y + plane.C * point.Z + plane.D);
  const auto denominator = std::sqrt(
      plane.A * plane.A + plane.B * plane.B + plane.C * plane.C);
  return numerator / denominator;
}

int CountInliers(
    const Plane& plane,
    const PointCloud& cloud,
    double threshold) {
  return std::count_if(cloud.begin(), cloud.end(),
      [&plane, threshold](const Point3d& point) {
        return CalcDistance(plane, point) <= threshold; });
}

Plane FindPlaneWithFullSearch(const PointCloud& cloud, double threshold) {
  const size_t cloud_size = cloud.size();

  Plane best_plane;
  int best_number_of_inliers = 0;

  for (size_t i = 0; i < cloud_size - 2; ++i) {
    for (size_t j = i + 1; j < cloud_size - 1; ++j) {
      for (size_t k = j + 1; k < cloud_size; ++k) {
        Plane sample_plane;
        if (!CreatePlane(sample_plane, cloud[i], cloud[j], cloud[k])) {
          continue;
        }

        const int number_of_inliers = CountInliers(
            sample_plane, cloud, threshold);
        if (number_of_inliers > best_number_of_inliers) {
          best_plane = sample_plane;
          best_number_of_inliers = number_of_inliers;
        }
      }
    }
  }

  return best_plane;
}

Plane FindPlaneWithSimpleRansac(
    const PointCloud& cloud,
    double threshold,
    size_t iterations) {
  const size_t cloud_size = cloud.size();

  // use uint64_t to calculate the number of all combinations
  // for N <= 25000 without overflow
  const auto cloud_size_ull = static_cast<uint64_t>(cloud_size);
  const auto number_of_combinations =
      cloud_size_ull * (cloud_size_ull - 1) * (cloud_size_ull - 2) / 6;

  if (number_of_combinations <= iterations) {
    return FindPlaneWithFullSearch(cloud, threshold);
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> distr(0, cloud_size - 1);

  Plane best_plane;
  int best_number_of_inliers = 0;

  for (size_t i = 0; i < iterations; ++i) {
    std::array<size_t, 3> indices;
    do {
      indices = {distr(gen), distr(gen), distr(gen)};
      std::sort(indices.begin(), indices.end());
    } while (indices[0] == indices[1] || indices[1] == indices[2]);

    Plane sample_plane;
    if (!CreatePlane(sample_plane,
                     cloud[indices[0]],
                     cloud[indices[1]],
                     cloud[indices[2]])) {
      continue;
    }

    const int number_of_inliers = CountInliers(
        sample_plane, cloud, threshold);
    if (number_of_inliers > best_number_of_inliers) {
      best_plane = sample_plane;
      best_number_of_inliers = number_of_inliers;
    }
  }

  return best_plane;
}

int main() {
  double p = 0.0;
  std::cin >> p;

  size_t points_number = 0;
  std::cin >> points_number;

  PointCloud cloud;
  cloud.reserve(points_number);
  for (size_t i = 0; i < points_number; ++i) {
    Point3d point;
    std::cin >> point.X >> point.Y >> point.Z;
    cloud.push_back(point);
  }

  const Plane plane = FindPlaneWithSimpleRansac(cloud, p, 100);

  std::cout << plane.A << ' '
            << plane.B << ' '
            << plane.C << ' '
            << plane.D << std::endl;

  return 0;
}
