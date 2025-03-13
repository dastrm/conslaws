#include "tests.hpp"

TEST(Numdiff, Diagonal_linear) {
  constexpr double a = 2, b = -3;
  constexpr auto f = [&](const double &x) { return a * x + b; };
  constexpr auto df = [&](const double & /*x*/) { return a; };

  const auto F = [&](const Eigen::VectorXd &x) { return x.unaryExpr(f); };
  const auto DF = [&](const Eigen::VectorXd &x) {
    return x.unaryExpr(df).asDiagonal().toDenseMatrix();
  };

  for (std::size_t n = 1; n <= 64; n *= 2) {
    const Eigen::VectorXd x = Eigen::VectorXd::Random(n);
    EXPECT_LT((numdiff(F, x) - DF(x)).lpNorm<Eigen::Infinity>(), 1e-7);
  }
}

TEST(Numdiff, Diagonal_sin) {
  constexpr auto f = [](const double &x) { return std::sin(x); };
  constexpr auto df = [](const double &x) { return std::cos(x); };

  const auto F = [&](const Eigen::VectorXd &x) { return x.unaryExpr(f); };
  const auto DF = [&](const Eigen::VectorXd &x) {
    return x.unaryExpr(df).asDiagonal().toDenseMatrix();
  };

  for (std::size_t n = 1; n <= 64; n *= 2) {
    const Eigen::VectorXd x = Eigen::VectorXd::Random(n);
    EXPECT_LT((numdiff(F, x) - DF(x)).lpNorm<Eigen::Infinity>(), 1e-7);
  }
}

TEST(Numdiff, Matrix_2x2) {
  const auto F = [](const Eigen::Vector2d &x) {
    Eigen::Vector2d y;
    y << 2 * x(0) + x(1), -x(0) * x(1);
    return y;
  };
  const auto DF = [](const Eigen::Vector2d &x) {
    Eigen::Matrix2d J;
    J << 2, -x(1), 1, -x(0);
    return J;
  };

  const Eigen::VectorXd x = Eigen::VectorXd::Random(2);
  EXPECT_LT((numdiff(F, x) - DF(x)).lpNorm<Eigen::Infinity>(), 1e-7);
}

TEST(Numdiff_sparse, Diagonal_linear) {
  constexpr double a = 2, b = -3;
  constexpr auto f = [&](const double &x) { return a * x + b; };
  constexpr auto df = [&](const double & /*x*/) { return a; };

  const auto F = [&](const Eigen::VectorXd &x) { return x.unaryExpr(f); };
  const auto DF = [&](const Eigen::VectorXd &x) -> Eigen::SparseMatrix<double> {
    return x.unaryExpr(df).asDiagonal().toDenseMatrix().sparseView();
  };

  for (std::size_t n = 1; n <= 64; n *= 2) {
    const Eigen::VectorXd x = Eigen::VectorXd::Random(n);
    EXPECT_LT((numdiff_sparse(F, x) - DF(x)).norm(), 1e-7);
  }
}

TEST(Numdiff_sparse, Diagonal_sin) {
  constexpr auto f = [](const double &x) { return std::sin(x); };
  constexpr auto df = [](const double &x) { return std::cos(x); };

  const auto F = [&](const Eigen::VectorXd &x) { return x.unaryExpr(f); };
  const auto DF = [&](const Eigen::VectorXd &x) -> Eigen::SparseMatrix<double> {
    return x.unaryExpr(df).asDiagonal().toDenseMatrix().sparseView();
  };

  for (std::size_t n = 1; n <= 64; n *= 2) {
    const Eigen::VectorXd x = Eigen::VectorXd::Random(n);
    EXPECT_LT((numdiff_sparse(F, x) - DF(x)).norm(), 1e-7);
  }
}

TEST(Numdiff_sparse, Matrix_2x2) {
  const auto F = [](const Eigen::Vector2d &x) {
    Eigen::Vector2d y;
    y << 2 * x(0) + x(1), -x(0) * x(1);
    return y;
  };
  const auto DF = [](const Eigen::Vector2d &x) -> Eigen::SparseMatrix<double> {
    Eigen::Matrix2d J;
    J << 2, -x(1), 1, -x(0);
    return J.sparseView();
  };

  const Eigen::VectorXd x = Eigen::VectorXd::Random(2);
  EXPECT_LT((numdiff_sparse(F, x) - DF(x)).norm(), 1e-7);
}

TEST(Quadrature, Gauss_Legendre) {
  constexpr auto n = [](const std::size_t &p) {
    return static_cast<std::size_t>(std::ceil((p + 1) / 2.0));
  };

  // unit interval
  constexpr auto f1 = [](const double &) { return 1.0; };
  EXPECT_NEAR(gauss_legendre<3>(f1, 0.0, 1.0), 1.0, 1e-12);

  // interval of length 3
  constexpr auto f2 = [](const double &) { return 1.3; };
  EXPECT_NEAR(gauss_legendre<1>(f2, 0.0, 3.0), 3 * 1.3, 1e-12);

  // shifted intervals
  EXPECT_NEAR(gauss_legendre<2>(f2, 0.0, 3.0),
              -gauss_legendre<2>(f2, 0.0, -3.0), 1e-12);
  EXPECT_NEAR(gauss_legendre<4>(f2, -12.0, -9.0),
              gauss_legendre<4>(f2, 2.0, 5.0), 1e-12);

  // third degree polynomial
  constexpr auto f3 = [](const double &x) { return std::pow(x, 3) - x + 1; };
  EXPECT_NEAR(gauss_legendre<n(3)>(f3, -0.3, 0.5), 0.7336, 1e-12);

  // fourth degree polynomial
  constexpr auto f4 = [](const double &x) {
    return -2 * std::pow(x, 4) + 3 * std::pow(x, 2) + x - 5;
  };
  EXPECT_NEAR(gauss_legendre<n(4)>(f4, -2.1, 2.75), -71.862825875, 1e-12);

  // trigonometric function
  constexpr auto f5 = [](const double &x) { return std::sin(x); };
  EXPECT_NEAR(gauss_legendre<10>(f5, -1.0, 0.0), std::cos(1.0) - 1, 1e-12);

  // impossible with symbolics
  constexpr auto f6 = [](const double &x) { return std::pow(x, x); };
  EXPECT_NEAR(gauss_legendre<32>(f6, 0.5, 1.0), 0.4108156482543906, 1e-12);
}

TEST(Quadrature, Simpsons1_3) {
  // unit interval
  constexpr auto f1 = [](const double &) { return 1.0; };
  EXPECT_NEAR(simpsons1_3(f1, 0.0, 1.0), 1.0, 1e-12);

  // interval of length 3
  constexpr auto f2 = [](const double &) { return 1.3; };
  EXPECT_NEAR(simpsons1_3(f2, 0.0, 3.0), 3 * 1.3, 1e-12);

  // shifted intervals
  EXPECT_NEAR(simpsons1_3(f2, 0.0, 3.0), -simpsons1_3(f2, 0.0, -3.0), 1e-12);
  EXPECT_NEAR(simpsons1_3(f2, -12.0, -9.0), simpsons1_3(f2, 2.0, 5.0), 1e-12);

  // third degree polynomial
  constexpr auto f3 = [](const double &x) { return std::pow(x, 3) - x + 1; };
  EXPECT_NEAR(simpsons1_3(f3, -0.3, 0.5), 0.7336, 1e-12);

  // fourth degree polynomial
  constexpr auto f4 = [](const double &x) {
    return -2 * std::pow(x, 4) + 3 * std::pow(x, 2) + x - 5;
  };
  EXPECT_GT(std::abs(simpsons1_3(f4, -2.1, 2.75) + 71.862825875), 2e1);

  // trigonometric function
  constexpr auto f5 = [](const double &x) { return std::sin(x); };
  EXPECT_NEAR(simpsons1_3(f5, -1.0, 0.0), std::cos(1.0) - 1, 1e-3);

  // impossible with symbolics
  constexpr auto f6 = [](const double &x) { return std::pow(x, x); };
  EXPECT_NEAR(simpsons1_3(f6, 0.5, 1.0), 0.4108156482543906, 1e-3);
}

TEST(Quadrature, Simpsons3_8) {
  // unit interval
  constexpr auto f1 = [](const double &) { return 1.0; };
  EXPECT_NEAR(simpsons3_8(f1, 0.0, 1.0), 1.0, 1e-12);

  // interval of length 3
  constexpr auto f2 = [](const double &) { return 1.3; };
  EXPECT_NEAR(simpsons3_8(f2, 0.0, 3.0), 3 * 1.3, 1e-12);

  // shifted intervals
  EXPECT_NEAR(simpsons3_8(f2, 0.0, 3.0), -simpsons3_8(f2, 0.0, -3.0), 1e-12);
  EXPECT_NEAR(simpsons3_8(f2, -12.0, -9.0), simpsons3_8(f2, 2.0, 5.0), 1e-12);

  // third degree polynomial
  constexpr auto f3 = [](const double &x) { return std::pow(x, 3) - x + 1; };
  EXPECT_NEAR(simpsons3_8(f3, -0.3, 0.5), 0.7336, 1e-12);

  // fourth degree polynomial
  constexpr auto f4 = [](const double &x) {
    return -2 * std::pow(x, 4) + 3 * std::pow(x, 2) + x - 5;
  };
  EXPECT_LT(std::abs(simpsons3_8(f4, -2.1, 2.75) + 71.862825875), 2e1);

  // trigonometric function
  constexpr auto f5 = [](const double &x) { return std::sin(x); };
  EXPECT_NEAR(simpsons3_8(f5, -1.0, 0.0), std::cos(1.0) - 1, 1e-3);

  // impossible with symbolics
  constexpr auto f6 = [](const double &x) { return std::pow(x, x); };
  EXPECT_NEAR(simpsons3_8(f6, 0.5, 1.0), 0.4108156482543906, 1e-3);
}

TEST(Damped_Newton, Linear) {
  constexpr double a = 3, b = -2;
  constexpr auto f = [&](const double &x) { return a * x + b; };
  constexpr auto df = [&](const double & /*x*/) { return a; };

  const auto F = [&](const Eigen::VectorXd &x) { return x.unaryExpr(f); };
  const auto DF = [&](const Eigen::VectorXd &x) {
    return x.unaryExpr(df).asDiagonal().toDenseMatrix();
  };

  constexpr double atol = 1e-10;
  constexpr double rtol = atol;

  for (std::size_t n = 1; n <= 64; n *= 2) {
    Eigen::VectorXd x = Eigen::VectorXd::Random(n);
    EXPECT_TRUE(damped_newton<double>(F, DF, x, rtol, atol, false));
    EXPECT_LT(
        (x - Eigen::VectorXd::Constant(n, -b / a)).lpNorm<Eigen::Infinity>(),
        atol);
  }
}

TEST(Damped_Newton, Exponential) {
  constexpr auto f = [&](const double &x) { return std::exp(x) - 1; };
  constexpr auto df = [&](const double &x) { return std::exp(x); };

  const auto F = [&](const Eigen::VectorXd &x) { return x.unaryExpr(f); };
  const auto DF = [&](const Eigen::VectorXd &x) {
    return x.unaryExpr(df).asDiagonal().toDenseMatrix();
  };

  constexpr double atol = 1e-10;
  constexpr double rtol = atol;

  for (std::size_t n = 1; n <= 64; n *= 2) {
    Eigen::VectorXd x = Eigen::VectorXd::Random(n);
    EXPECT_TRUE(damped_newton<double>(F, DF, x, rtol, atol, false));
    EXPECT_LT(x.lpNorm<Eigen::Infinity>(), atol);
  }
}

TEST(Damped_Newton, Infeasible) {
  constexpr auto f = [&](const double &x) { return std::sin(x) + 2; };
  constexpr auto df = [&](const double &x) { return std::cos(x); };

  const auto F = [&](const Eigen::VectorXd &x) { return x.unaryExpr(f); };
  const auto DF = [&](const Eigen::VectorXd &x) {
    return x.unaryExpr(df).asDiagonal().toDenseMatrix();
  };

  constexpr double atol = 1e-10;
  constexpr double rtol = atol;

  for (std::size_t n = 1; n <= 64; n *= 2) {
    Eigen::VectorXd x = Eigen::VectorXd::Random(n);
    EXPECT_FALSE(damped_newton<double>(F, DF, x, rtol, atol, false));
  }
}

TEST(Damped_Newton_sparse, Linear) {
  constexpr double a = 3, b = -2;
  constexpr auto f = [&](const double &x) { return a * x + b; };
  constexpr auto df = [&](const double & /*x*/) { return a; };

  const auto F = [&](const Eigen::VectorXd &x) { return x.unaryExpr(f); };
  const auto DF = [&](const Eigen::VectorXd &x) {
    const std::size_t n = x.size();
    const Eigen::VectorXd dx = x.unaryExpr(df);
    Eigen::SparseMatrix<double> J(n, n);
    J.reserve(n);
    for (std::size_t i = 0; i < n; ++i) J.insert(i, i) = dx(i);
    return J;
  };

  constexpr double atol = 1e-10;
  constexpr double rtol = atol;

  for (std::size_t n = 1; n <= 64 * 16; n *= 2) {
    Eigen::VectorXd x = Eigen::VectorXd::Random(n);
    EXPECT_TRUE(damped_newton_sparse<double>(F, DF, x, rtol, atol, false));
    EXPECT_LT(
        (x - Eigen::VectorXd::Constant(n, -b / a)).lpNorm<Eigen::Infinity>(),
        atol);
  }
}

TEST(Damped_Newton_sparse, Exponential) {
  constexpr auto f = [&](const double &x) { return std::exp(x) - 1; };
  constexpr auto df = [&](const double &x) { return std::exp(x); };

  const auto F = [&](const Eigen::VectorXd &x) { return x.unaryExpr(f); };
  const auto DF = [&](const Eigen::VectorXd &x) -> Eigen::SparseMatrix<double> {
    return x.unaryExpr(df).asDiagonal().toDenseMatrix().sparseView();
  };

  constexpr double atol = 1e-10;
  constexpr double rtol = atol;

  for (std::size_t n = 1; n <= 64; n *= 2) {
    Eigen::VectorXd x = Eigen::VectorXd::Random(n);
    EXPECT_TRUE(damped_newton_sparse<double>(F, DF, x, rtol, atol, false));
    EXPECT_LT(x.lpNorm<Eigen::Infinity>(), atol);
  }
}

TEST(Damped_Newton_sparse, Infeasible) {
  constexpr auto f = [&](const double &x) { return std::sin(x) + 2; };
  constexpr auto df = [&](const double &x) { return std::cos(x); };

  const auto F = [&](const Eigen::VectorXd &x) { return x.unaryExpr(f); };
  const auto DF = [&](const Eigen::VectorXd &x) -> Eigen::SparseMatrix<double> {
    return x.unaryExpr(df).asDiagonal().toDenseMatrix().sparseView();
  };

  constexpr double atol = 1e-10;
  constexpr double rtol = atol;

  for (std::size_t n = 1; n <= 64; n *= 2) {
    Eigen::VectorXd x = Eigen::VectorXd::Random(n);
    EXPECT_FALSE(damped_newton_sparse<double>(F, DF, x, rtol, atol, false));
  }
}

TEST(Linear_transport, Smooth_forward) {
  static const std::vector<std::size_t> rhs_nums = {1, 2, 3};
  static const std::vector<std::size_t> rk_nums = {1, 3, 3};
  static const std::vector<double> order_mins = {0.9, 1.9, 0.9};
  constexpr std::size_t n_min = 0;
  constexpr std::size_t n_max = 9;

  constexpr double x_min = -1.3;
  constexpr double x_max = M_PI;
  constexpr double t_end = 1.2 * std::sqrt(3);
  constexpr double cfl_number = 0.95;
  constexpr auto ICs = [x_min, x_max](const double &x) {
    return std::sin(2 * M_PI * x / (x_max - x_min));
  };

  constexpr auto Flux = [](const double &u) { return u; };
  constexpr auto DFlux = [](const double & /*u*/) { return 1.0; };
  static const auto Exact = [&](const double &x) {
    return ICs(x - DFlux(x) * t_end);
  };

  test_convergence(rhs_nums, rk_nums, order_mins, n_min, n_max, x_min, x_max,
                   t_end, cfl_number, ICs, Exact, Flux, DFlux);
}

TEST(Linear_transport, Smooth_backward) {
  static const std::vector<std::size_t> rhs_nums = {1, 2, 3};
  static const std::vector<std::size_t> rk_nums = {1, 3, 3};
  static const std::vector<double> order_mins = {0.9, 1.9, 0.9};
  constexpr std::size_t n_min = 0;
  constexpr std::size_t n_max = 9;

  constexpr double x_min = -11.2;
  constexpr double x_max = -5;
  constexpr double t_end = 0.81;
  constexpr double cfl_number = std::sqrt(2) / 3;
  constexpr auto ICs = [x_min, x_max](const double &x) {
    return std::cos(2 * M_PI * x / (x_max - x_min));
  };

  constexpr auto Flux = [](const double &u) { return -u; };
  constexpr auto DFlux = [](const double & /*u*/) { return -1.0; };
  static const auto Exact = [&](const double &x) {
    return ICs(x - DFlux(x) * t_end);
  };

  test_convergence(rhs_nums, rk_nums, order_mins, n_min, n_max, x_min, x_max,
                   t_end, cfl_number, ICs, Exact, Flux, DFlux);
}
