#pragma once

#include <ATen/ATen.h>

#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#include <tuple>

namespace express {
struct Variable;
struct Engine;

using varptr = std::shared_ptr<Variable>;
using const_varptr = std::shared_ptr<const Variable>;
using varptr_list = std::vector< std::shared_ptr<Variable>>;
using const_varptr_list = std::vector< std::shared_ptr<const Variable>>;
using tensor_list = std::vector<at::Tensor>;
using vargrad_list = std::vector< std::tuple<varptr, at::Tensor>>;

struct Operator : std::enable_shared_from_this<Operator> {
public:
  Operator() = default;
  virtual ~Operator() {
      std::cout << " op is deleted" << std::endl;
  }

  /// Operators are neither copyable nor moveable.
  Operator(const Operator& other) = delete;
  Operator(Operator&& other) = delete;
  Operator& operator=(const Operator& other) = delete;
  Operator& operator=(Operator&& other) = delete;

  std::shared_ptr<Operator> get_shared_ptr();

  varptr_list forward(const const_varptr_list& bottoms);
  vargrad_list backward(const tensor_list& grads, std::vector<int> outputs);

protected:
  virtual varptr_list _forward(const const_varptr_list& bottoms) = 0;
  virtual tensor_list _backward(const tensor_list& grads) = 0;

protected:
  std::vector< std::shared_ptr<Variable>> bottoms_;
  std::vector<at::Tensor> tops_;
};

inline std::shared_ptr<Operator> Operator::get_shared_ptr() {
  return shared_from_this();
}

struct AccumulatedOperator : Operator {
public:
    AccumulatedOperator() = default;
    virtual ~AccumulatedOperator() = default;

    static inline std::shared_ptr<AccumulatedOperator> singleton() {
        if ( singleton_.get() == nullptr) {
            singleton_ = std::make_shared<AccumulatedOperator>();
        }
        return singleton_;
    }
protected:
    virtual varptr_list _forward(const const_varptr_list& bottoms) override;
    virtual tensor_list _backward(const tensor_list& grads) override;

private:
    static std::shared_ptr<AccumulatedOperator> singleton_;
};


} // namespace express
