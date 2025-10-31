#include <taskflow/taskflow.hpp>  // Taskflow header

int main() {
    tf::Executor executor;
    tf::Taskflow taskflow("Condition Task Demo");

    int counter = 0;
    const int limit = 5;

    auto init = taskflow.emplace([&]() {
        printf("Initialize counter = %d\n", counter);
    });

    auto loop = taskflow.emplace([&]() {
        printf("Loop iteration %d\n", counter);
        counter++;
        return (counter < limit) ? 0 : 1;  // 0 => loop again, 1 => exit
    }).condition();

    auto done = taskflow.emplace([]() {
        printf("Loop done.\n");
    });

    init.precede(loop);
    loop.precede(loop, done);  // self-edge enables iteration

    executor.run(taskflow).wait();
    return 0;
}
