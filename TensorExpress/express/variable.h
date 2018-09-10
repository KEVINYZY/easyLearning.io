#pragma once

#include <ATen/ATen.h>

#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace express {
struct Operator;

struct Variable : std::enable_shared_from_this<Variable> {
public:
  Variable() = default;
  virtual ~Variable() = default;

  // basic control
  const char* toString() const;
  const at::Tensor& data() const noexcept;
  at::Tensor& data() noexcept;
  void set_name(const std::string& name);
  const std::string& name() const noexcept;
  std::shared_ptr<Variable> get_shared_ptr();
  std::shared_ptr<const Operator> grad_op() const;
  const int grad_output() const;

  // fast access variable's properties
  at::IntList sizes() const;
  int64_t dim() const;
  at::ScalarType scalarType() const;
  at::Backend backend() const;

  // fast inline check
  bool isSymbol() const;
  bool isVariable() const;
  bool isAccumulated() const;
  bool isConstant() const;
  bool isReacted() const;

protected:
  std::string name_;
  std::mutex mutex_;

  at::Tensor data_;
  at::Tensor grad_;

  std::shared_ptr<Operator> grad_op_;
  int grad_output_;

  friend std::shared_ptr<Variable> make_variable(at::Tensor data, std::shared_ptr<Operator> grad_op, int grad_output);
  friend std::shared_ptr<Variable> make_variable(at::Tensor data, bool requires_grad);
  friend std::shared_ptr<Variable> make_symbol(bool requires_grad);
  friend std::shared_ptr<Variable> make_symbol(std::shared_ptr<Operator> grad_op, int grad_output);

  friend int backward(std::shared_ptr<Variable> var, std::shared_ptr<const Variable> grad);
};

inline const char* Variable::toString() const {
    if ( isSymbol() ) {
        return "Symbol";
    }
    return "Variable";
}

inline const at::Tensor& Variable::data() const noexcept {
  return data_;
}

inline at::Tensor& Variable::data() noexcept {
  return data_;
}

inline void Variable::set_name (const std::string& name) {
  name_ = name;
}

inline const std::string& Variable::name() const noexcept {
  return name_;
}

inline std::shared_ptr<Variable> Variable::get_shared_ptr() {
  return shared_from_this();
}

inline std::shared_ptr<const Operator> Variable::grad_op() const {
  return grad_op_;
}

inline const int  Variable::grad_output() const {
  return grad_output_;
}

inline at::IntList Variable::sizes() const {
  return data_.sizes();
}

inline int64_t Variable::dim() const {
  return data_.dim();
}

inline at::ScalarType Variable::scalarType() const {
  return data_.pImpl->type().scalarType();
}

inline at::Backend Variable::backend() const {
  return data_.pImpl->type().backend();
}

std::shared_ptr<Variable> make_variable(at::Tensor data, std::shared_ptr<Operator> grad_op, int grad_output = 0);
std::shared_ptr<Variable> make_variable(at::Tensor data, bool requires_grad);
std::shared_ptr<Variable> make_symbol();
std::shared_ptr<Variable> make_symbol(std::shared_ptr<Operator> grad_op, int grad_output = 0);

} // namespace express
