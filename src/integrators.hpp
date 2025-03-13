#ifndef INTEGRATORS_HPP
#define INTEGRATORS_HPP

#include "utils.hpp"

// Butcher tableau based Runge-Kutta explicit solver for autonomous ODEs
template <typename T>
class ExplRKIntegrator {
public:
  ExplRKIntegrator(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &A,
                   const Eigen::Matrix<T, Eigen::Dynamic, 1> &b)
      : A_(A), b_(b), s_(b.size()) {
    verify(A.cols() == A.rows(), "Matrix must be square.");
    verify(A.cols() == b.size(), "Incompatible matrix/vector size.");
  }

  template <typename F1, typename F2, typename F3>
  Eigen::Matrix<T, Eigen::Dynamic, 1>
  solve(F1 &&rhs, F2 && /*Drhs*/, const std::size_t & /*rhs_num*/,
        const Eigen::Matrix<T, Eigen::Dynamic, 1> &u, const T &t_end,
        const T &cfl_times_h, F3 &&DFlux) const {
    const std::size_t d = u.size();
    Eigen::Matrix<T, Eigen::Dynamic, 1> mu = u;
    Eigen::Matrix<T, Eigen::Dynamic, 1> incr(d);
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> K(d, s_);

    T t = T(0);
    while (t != t_end) {
      const T max_char_speed = mu.unaryExpr(DFlux).cwiseAbs().maxCoeff();
      const T dt = std::min(cfl_times_h / max_char_speed, t_end - t);

      K.col(0) = rhs(mu);
      for (std::size_t i = 1; i < s_; ++i) {
        incr.setZero();
        for (std::size_t j = 0; j < i; ++j) {
          incr += A_(i, j) * K.col(j);
        }
        K.col(i) = rhs(mu + dt * incr);
      }
      mu += dt * K * b_;
      t += dt;
    }

    return mu;
  }

private:
  // Butcher data
  const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A_;
  const Eigen::Matrix<T, Eigen::Dynamic, 1> b_;
  const std::size_t s_; // size of Butcher tableau
};

// Butcher tableau based Runge-Kutta implicit solver for autonomous ODEs
template <typename T>
class ImplRKIntegrator {
public:
  ImplRKIntegrator(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &A,
                   const Eigen::Matrix<T, Eigen::Dynamic, 1> &b)
      : A_(A), b_(b), s_(b.size()) {
    verify(A.cols() == A.rows(), "Matrix must be square.");
    verify(A.cols() == b.size(), "Incompatible matrix/vector size.");
  }

  template <typename F1, typename F2, typename F3>
  Eigen::Matrix<T, Eigen::Dynamic, 1>
  solve(F1 &&rhs, F2 &&Drhs, const std::size_t &rhs_num,
        const Eigen::Matrix<T, Eigen::Dynamic, 1> &u, const T &t_end,
        const T &cfl_times_h, F3 &&DFlux) const {
    const std::size_t d = u.size();
    Eigen::Matrix<T, Eigen::Dynamic, 1> mu = u;
    Eigen::Matrix<T, Eigen::Dynamic, 1> g(s_ * d);
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> K(d, s_);
    const Eigen::SparseMatrix<T> Eye =
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Identity(s_ * d,
                                                                   s_ * d)
            .sparseView();

    std::vector<Eigen::SparseMatrix<T>> A_kron_I;
    A_kron_I.reserve(s_);
    for (std::size_t j = 0; j < s_; ++j)
      A_kron_I.push_back(Eigen::KroneckerProductSparse(
          A_.col(j),
          Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Identity(d, d)
              .sparseView()));

    T t = T(0);
    while (t != t_end) {
      const T max_char_speed = mu.unaryExpr(DFlux).cwiseAbs().maxCoeff();
      const T dt = std::min(cfl_times_h / max_char_speed, t_end - t);

      static const auto F = [&](const Eigen::Matrix<T, Eigen::Dynamic, 1> &g_) {
        Eigen::Matrix<T, Eigen::Dynamic, 1> F_ = g_;
        for (std::size_t j = 0; j < s_; ++j) {
          F_ -= dt * A_kron_I[j] * rhs(mu + g_.segment(j * d, d));
        }
        return F_;
      };

      static const auto DF = [&](const Eigen::Matrix<T, Eigen::Dynamic, 1> &g_)
          -> Eigen::SparseMatrix<T> {
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> DF_(s_ * d, s_ * d);
        for (std::size_t j = 0; j < s_; ++j) {
          DF_.block(0, j * d, s_ * d, d) =
              A_kron_I[j] * Drhs(mu + g_.segment(j * d, d));
        }
        return Eye - dt * DF_.sparseView();
      };

      static const auto DF_numdiff =
          [&](const Eigen::Matrix<T, Eigen::Dynamic, 1> &g_)
          -> Eigen::SparseMatrix<T> {
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> DF_(s_ * d, s_ * d);
        for (std::size_t j = 0; j < s_; ++j) {
          DF_.block(0, j * d, s_ * d, d) =
              A_kron_I[j] *
              numdiff_sparse(rhs, (mu + g_.segment(j * d, d)).eval());
        }
        return Eye - dt * DF_.sparseView();
      };

      g.setZero();
#if 1
      if (rhs_num == 1)
        damped_newton_sparse<T>(F, DF, g);
      else
        damped_newton_sparse<T>(F, DF_numdiff, g);
#else
      // TODO: adaptive option
      while (!damped_newton_sparse<T>(F, DF_numdiff, g)) {
        // g.setRandom();
        dt *= 0.5;
        std::cout << "  new dt: " << dt << std::endl;
      }
      dt *= 1.25;
#endif

      for (std::size_t j = 0; j < s_; ++j)
        K.col(j) = rhs(mu + g.segment(j * d, d));
      mu += dt * K * b_;
      t += dt;
    }

    return mu;
  }

private:
  // Butcher data
  const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A_;
  const Eigen::Matrix<T, Eigen::Dynamic, 1> b_;
  const std::size_t s_; // size of Butcher tableau
};

// Butcher tableau based Runge-Kutta diagonally implicit solver for autonomous
// ODEs
template <typename T>
class DiagImplRKIntegrator {
public:
  DiagImplRKIntegrator(
      const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &A,
      const Eigen::Matrix<T, Eigen::Dynamic, 1> &b)
      : A_(A), b_(b), s_(b.size()) {
    verify(A.cols() == A.rows(), "Matrix must be square.");
    verify(A.cols() == b.size(), "Incompatible matrix/vector size.");
  }

  template <typename F1, typename F2, typename F3>
  Eigen::Matrix<T, Eigen::Dynamic, 1>
  solve(F1 &&rhs, F2 &&Drhs, const std::size_t &rhs_num,
        const Eigen::Matrix<T, Eigen::Dynamic, 1> &u, const T &t_end,
        const T &cfl_times_h, F3 &&DFlux) const {
    const std::size_t d = u.size();
    Eigen::Matrix<T, Eigen::Dynamic, 1> mu = u;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> K(d, s_), G(d, s_);
    const Eigen::SparseMatrix<T> Eye =
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Identity(d, d)
            .sparseView();

    T t = T(0);
    while (t != t_end) {
      const T max_char_speed = mu.unaryExpr(DFlux).cwiseAbs().maxCoeff();
      const T dt = std::min(cfl_times_h / max_char_speed, t_end - t);

      G.setZero();
      for (std::size_t i = 0; i < s_; ++i) {
        static const auto F =
            [&](const Eigen::Matrix<T, Eigen::Dynamic, 1> &g) {
              Eigen::Matrix<T, Eigen::Dynamic, 1> F_ = g;
              for (std::size_t j = 0; j < i; ++j)
                F_ -= dt * A_(i, j) * rhs(mu + G.col(j));
              F_ -= dt * A_(i, i) * rhs(mu + g);
              return F_;
            };

        static const auto DF = [&](const Eigen::Matrix<T, Eigen::Dynamic, 1> &g)
            -> Eigen::SparseMatrix<T> {
          return Eye - dt * A_(i, i) * Drhs(mu + g);
        };

        static const auto DF_numdiff =
            [&](const Eigen::Matrix<T, Eigen::Dynamic, 1> &g)
            -> Eigen::SparseMatrix<T> {
          return Eye - dt * A_(i, i) * numdiff_sparse(rhs, (mu + g).eval());
        };

        if (rhs_num == 1)
          damped_newton_sparse<T>(F, DF, G.col(i));
        else
          damped_newton_sparse<T>(F, DF_numdiff, G.col(i));
      }

      for (std::size_t j = 0; j < s_; ++j) K.col(j) = rhs(mu + G.col(j));
      mu += dt * K * b_;
      t += dt;
    }

    return mu;
  }

private:
  // Butcher data
  const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A_;
  const Eigen::Matrix<T, Eigen::Dynamic, 1> b_;
  const std::size_t s_; // size of Butcher tableau
};

// Butcher tableau based Runge-Kutta semi-implicit (linearly implicit) solver
// (Rosenbrock-Wanner method) for linear autonomous ODEs
template <typename T>
class SemiImplRKIntegrator {
public:
  // Constructor for the RK method.
  SemiImplRKIntegrator(
      const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &A,
      const Eigen::Matrix<T, Eigen::Dynamic, 1> &b,
      const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &G)
      : A_(A), b_(b), G_(G), s_(b.size()) {
    verify(A.cols() == A.rows(), "Matrix must be square.");
    verify(G.cols() == G.rows(), "Matrix must be square.");
    verify(A.cols() == b.size(), "Incompatible matrix/vector size.");
    verify(G.cols() == b.size(), "Incompatible matrix/vector size.");

    // additionally, we require that all entries of the main diagonal of G are
    // equal, such that we need to invert only one coefficient matrix
    for (std::size_t i = 1; i < s_; ++i)
      verify(G(i, i) == G(i - 1, i - 1), "Incompatible main diagonal.");
  }

  template <typename F1, typename F2, typename F3>
  Eigen::Matrix<T, Eigen::Dynamic, 1>
  solve(F1 &&rhs, F2 &&Drhs, const std::size_t &rhs_num,
        const Eigen::Matrix<T, Eigen::Dynamic, 1> &u, const T &t_end,
        const T &cfl_times_h, F3 &&DFlux) const {
    const std::size_t d = u.size();
    Eigen::Matrix<T, Eigen::Dynamic, 1> mu = u;
    Eigen::Matrix<T, Eigen::Dynamic, 1> incr(d), incr2(d);
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> K(d, s_);
    const Eigen::SparseMatrix<T> Eye =
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Identity(d, d)
            .sparseView();
    Eigen::SparseMatrix<T> J;
    if (rhs_num == 1)
      J = Drhs(mu);
    else
      J = numdiff_sparse(rhs, mu);
    Eigen::SparseLU<Eigen::SparseMatrix<T>> solver;
    const T max_char_speed = mu.unaryExpr(DFlux).cwiseAbs().maxCoeff();

    T t = T(0), dt_old = T(0);
    while (t != t_end) {
      const T dt = std::min(cfl_times_h / max_char_speed, t_end - t);
      if (dt_old != dt) {
        solver.compute(Eye - dt * G_(0, 0) * J);
        dt_old = dt;
      }

      K.col(0) = solver.solve(rhs(mu));
      for (std::size_t i = 1; i < s_; ++i) {
        incr.setZero();
        incr2.setZero();
        for (std::size_t j = 0; j < i; ++j) {
          incr += (G_(i, j) + A_(i, j)) * K.col(j);
          incr2 += A_(i, j) * K.col(j);
        }
        incr = rhs(mu + dt * incr) - dt * J * incr2;
        K.col(i) = solver.solve(incr);
      }
      mu += dt * K * b_;
      t += dt;
    }

    return mu;
  }

private:
  // Butcher data
  const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A_;
  const Eigen::Matrix<T, Eigen::Dynamic, 1> b_;
  const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> G_;
  const std::size_t s_; // size of Butcher tableau
};

#endif /* INTEGRATORS_HPP */
