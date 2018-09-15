#include "variable.h"
#include "operator.h"

namespace express {

bool Variable::isSymbol() const {
  return data_.pImpl == at::UndefinedTensor::singleton();
}

bool Variable::isVariable() const {
  return data_.pImpl != at::UndefinedTensor::singleton();
}

bool Variable::isAccumulated() const {
  return grad_op_ == AccumulatedOperator::singleton();
}

bool Variable::isConstant() const {
  return grad_op_.get() == nullptr;
}

bool Variable::isReacted() const {
  return grad_op_.get() && ( grad_op_ != AccumulatedOperator::singleton() );
}

std::shared_ptr<Variable> make_variable(at::Tensor data, std::shared_ptr<Operator> grad_op, int grad_output) {
  auto var_ptr = std::make_shared<express::Variable>();
  var_ptr->data_ = data;
  var_ptr->grad_op_ = grad_op;
  var_ptr->grad_output_ = grad_output;
  return var_ptr;
}

std::shared_ptr<Variable> make_variable(at::Tensor data, bool requires_grad) {
  auto var_ptr = std::make_shared<express::Variable>();

  var_ptr->data_ = data;
  if ( requires_grad ) {
    var_ptr->grad_op_ = AccumulatedOperator::singleton();
    var_ptr->grad_output_ = 0;
    var_ptr->grad_ = at::zeros_like(data);
  } else {
    var_ptr->grad_op_.reset();
    var_ptr->grad_output_ = -1;
  }

  return var_ptr;
}

std::shared_ptr<Variable> make_symbol(bool requires_grad) {
  auto var_ptr = std::make_shared<express::Variable>();

  if ( requires_grad ) {
    var_ptr->grad_op_ = AccumulatedOperator::singleton();
    var_ptr->grad_output_ = 0;
  } else {
    var_ptr->grad_op_.reset();
    var_ptr->grad_output_ = -1;
  }

  return var_ptr;
}

std::shared_ptr<Variable> make_symbol(std::shared_ptr<Operator> grad_op, int grad_output){
  auto var_ptr = std::make_shared<express::Variable>();
  var_ptr->grad_op_ = grad_op;
  var_ptr->grad_output_ = grad_output;
  return var_ptr;
}

} // namespace express
