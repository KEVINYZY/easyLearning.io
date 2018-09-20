#include <rpc/server.h>
#include <rpc/client.h>
#include <unistd.h>
#include <iostream>

#include "express.h"

void testATen() {
    // testing basic tensor from ATen
    at::Tensor foo = at::CPU(at::kFloat).rand({12,12});

    {
        at::Tensor bar = foo;
        auto ac = bar.accessor<float,2>();
        ac[10][10] = 3.1415926;
    }

    auto ac = foo.accessor<float, 2>();
    std::cout << ac[10][10] << std::endl;
}

void testExpress() {
    at::Tensor foo = at::CPU(at::kFloat).rand({3,3});
    auto var = express::make_variable(foo, true);

    std::cout << var->toString() << std::endl;
    std::cout << var->scalarType() << std::endl;
    std::cout << var->backend() << std::endl;
    std::cout << var->isAccumulated() << std::endl;
    std::cout << var->isConstant() << std::endl;

    auto ac = var->data().accessor<float, 2>();
    ac[1][1] = -3.1415;

    auto var2 = express::api::abs(var);
    ac = var2->data().accessor<float, 2>();
    std::cout << ac[1][1] << std::endl;

    at::Tensor bar = at::CPU(at::kFloat).rand({3,3});
    auto grad = express::make_variable(bar, false);
    int result = express::backward(var2, grad);
    std::cout << "backward result = " << result << std::endl;
}

int main(const int argc, const char* argv[]) {
    testATen();
    testExpress();

    rpc::server srv(8080); // listen on TCP port 8080
    srv.bind("add", [](double a, double b) { return a + b; });

    srv.async_run(1);

    std::cout << " Run in client mode " << std::endl;
    rpc::client client("127.0.0.1", 8080);
    for(;;) {
        double five = client.call("add", 2, 3).as<double>();
        std::cout << "Get result " << five << std::endl;
        sleep(1);
    }
    return 0;
}
