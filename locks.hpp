#ifndef __LOCKS_H
#define __LOCKS_H

#ifdef USE_OPENMP_LOCK
#else
#ifdef USE_SPINLOCK 
#include <atomic>
std::atomic_flag lkd_ = ATOMIC_FLAG_INIT;
#else
#include <mutex>
std::mutex mtx_;
#endif
void lock() {
#ifdef USE_SPINLOCK 
    while (lkd_.test_and_set(std::memory_order_acquire)) { ; } 
#else
    mtx_.lock();
#endif
}
void unlock() { 
#ifdef USE_SPINLOCK 
    lkd_.clear(std::memory_order_release); 
#else
    mtx_.unlock();
#endif
}
#endif

#endif // __LOCKS_H
