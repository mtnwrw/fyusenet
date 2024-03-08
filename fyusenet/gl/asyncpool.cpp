//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Thread-Pool for Asynchronous (GL) Operations
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <cassert>
#include <chrono>
#include <cinttypes>

//-------------------------------------- Project  Headers ------------------------------------------

#include "glexception.h"
#include "asyncpool.h"
#include "../gpu/gfxcontextmanager.h"

//-------------------------------------- Global Variables ------------------------------------------

namespace fyusion {
namespace opengl {
//-------------------------------------- Local Definitions -----------------------------------------

std::list<AsyncPool::entry> AsyncPool::threads_;
std::list<AsyncPool::entry> AsyncPool::glThreads_;
std::mutex AsyncPool::lock_;
std::atomic<int> AsyncPool::teardownProtection_{0};
std::function<void()> AsyncPool::ThreadImpl::EMPTY;
std::thread * AsyncPool::watchDog_ = nullptr;
bool AsyncPool::goingDown_ = false;
int AsyncPool::maxGLThreads_ = 8;
uint64_t AsyncPool::waitCycles_ = 0;
uint64_t AsyncPool::immediateHit_ = 0;
uint64_t AsyncPool::requests_ = 0;
uint32_t AsyncPool::reapCount_ = 0;

/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @brief Retrieve non-context-associated thread for parallel execution
 *
 * @return Thread object which can be used immediately
 *
 * This function checks if an idle thread is available in the (non-context) thread-pool and returns
 * the next available instance. If no thread is available, a new thread is created.
 *
 * @warning This function has a <b>hard maximum</b> associated (which is #HARD_MAX_THREADS), when
 *          this is exceeded it will throw an exception to protect the system.
 */
AsyncPool::Thread AsyncPool::getThread() {
    std::lock_guard<std::mutex> lck(lock_);
    if (goingDown_) return Thread();
    teardownProtection_.fetch_add(1);
    if (!watchDog_) watchDog_ = new std::thread(sniff);
    for (auto it = threads_.begin(); it != threads_.end(); ++it) {
        uint32_t expected = 0;
        entry & ent = (*it);
        if (ent.refcount.compare_exchange_strong(expected, 1)) {
            assert(ent.refcount.load() == 1);
            if (!ent.thread->isBusy()) {
                teardownProtection_.fetch_sub(1);
                return Thread(ent.thread, &ent.refcount);
            } else ent.refcount.store(0);
        }
    }
    if ((threads_.size() + glThreads_.size()) >= HARD_MAX_THREADS) {
        THROW_EXCEPTION_ARGS(GLException, "Maximum number of threads (%d) reached and trying to create one more, something is not right", HARD_MAX_THREADS);
    }
    ThreadImpl* newthread = new ThreadImpl();
    threads_.emplace_back(newthread,1);
    entry & ent = threads_.back();
    assert(ent.thread == newthread);
    teardownProtection_.fetch_sub(1);
    return Thread(ent.thread, &ent.refcount);
}

void AsyncPool::setMaxGLThreads(int mt) {
    std::lock_guard<std::mutex> lck(lock_);
    maxGLThreads_ = mt;
}

void AsyncPool::createDerivedBatch(const fyusenet::GfxContextLink& ctx, int numThreads) {
    std::lock_guard<std::mutex> lck(lock_);
    if (!ctx.context_) THROW_EXCEPTION_ARGS(GLException,"No valid context supplied");
    assert(numThreads > 0);
    if (!goingDown_) {
        teardownProtection_.fetch_add(1);
        for (int i=0; i<numThreads; i++) {
            fyusenet::GfxContextLink newctx = fyusenet::GfxContextManager::instance()->createDerived(ctx);
            GLThreadImpl * newthread = new GLThreadImpl(newctx);
            glThreads_.emplace_back(newthread,0);
        }
        teardownProtection_.fetch_sub(1);
    }
}


/**
 * @brief Get a GL thread that uses a derived or shared context to the supplied one
 *
 * @param ctx Context to find a thread for that has a shared context
 * @param timeout Optional parameter, sets a timeout in milliseconds to attempt to get a thread,
 *                -1 (the default) waits forever
 *
 * @return GLThread instance that uses a context that is shared with the supplied context or an
 *         invalid thread if the pool is taken down or the timeout has been exceeded
 *
 * @warning This function has a <b>hard maximum</b> associated (which is #HARD_MAX_THREADS), when
 *          this is exceeded it will throw an exception to protect the system.
 */
AsyncPool::GLThread AsyncPool::getDerivedContextThread(const fyusenet::GfxContextLink& ctx, int timeout) {
    using namespace std::chrono_literals;
    if (!ctx.context_) THROW_EXCEPTION_ARGS(GLException,"No valid context supplied");
    lock_.lock();
    if (goingDown_) return GLThread();
    if (!watchDog_) watchDog_ = new std::thread(sniff);
    requests_++;
    teardownProtection_.fetch_add(1);        // we unlock in the meantime and this prevents a teardown while we are working on stuff
    bool done = ((int)glThreads_.size() < maxGLThreads_);
    bool immediate = true;
    int waittime = 0;
    do {
        for (auto it = glThreads_.begin(); it != glThreads_.end(); ++it) {
            entry & ent = (*it);
            fyusenet::GfxContextLink & c = (static_cast<GLThreadImpl *>(ent.thread))->context_;
            bool suitable = false;
            if (ctx.interface()->isDerived()) {
                suitable = c.interface()->isDerivedFrom(ctx.interface()->getMain());
            } else {
                suitable = c.interface()->isDerivedFrom(ctx.interface());
            }
            if (suitable) {
                uint32_t expected = 0;
                if (ent.refcount.compare_exchange_strong(expected, 1)) {
                    if (!ent.thread->isBusy()) {
                        teardownProtection_.fetch_sub(1);
                        if (immediate) immediateHit_++;
                        lock_.unlock();
                        return GLThread((GLThreadImpl *)ent.thread, &ent.refcount);
                    } else {
                        ent.refcount.store(0);
                    }
                }
            }
        }
        if (((int)glThreads_.size() >= maxGLThreads_) || ((int)glThreads_.size() + threads_.size() >= HARD_MAX_THREADS)) {
            waitCycles_++;
            lock_.unlock();
            std::this_thread::sleep_for(5ms);           // TODO (mw) get rid of the polling
            waittime += 5;
            if ((timeout >= 0) && (waittime >= timeout)) {
                return GLThread();
            }
            lock_.lock();
            immediate = false;
        } else done = true;
    } while (!done);
    if ((threads_.size() + glThreads_.size()) >= HARD_MAX_THREADS) {
        THROW_EXCEPTION_ARGS(GLException, "Maximum number of threads (%d) reached and trying to create one more, something is not right", HARD_MAX_THREADS);
    }
    auto mgr = fyusenet::GfxContextManager::instance();
    fyusenet::GfxContextLink newctx = mgr->createDerived(ctx.interface()->getMain());
    GLThreadImpl * newthread = new GLThreadImpl(newctx);
    glThreads_.emplace_back(newthread,1);
    entry & ent = glThreads_.back();
    teardownProtection_.fetch_sub(1);
    if (immediate) immediateHit_++;
    lock_.unlock();
    return GLThread((GLThreadImpl *)ent.thread, &ent.refcount);
}


/**
 * @brief Get a thread specific to an OpenGL context in a blocking manner
 *
 * @param ctx OpenGL context to obtain thread for
 * @param timeout Optional parameter, sets a timeout in milliseconds to attempt to get a thread,
 *                -1 (the default) waits forever
 *
 * @return Thread object that wraps the actual thread implementation or an invalid thread
 *         if the thread-pool is in cleanup / taken down of if the timeout has been exceeded
 *
 * @warning This function may block for a long time if the implementation of the task running
 *          in the thread is not efficient. We can only have \b one thread per GL context and
 *          hence, everything that executes in a context thread is <b>time-critical</b>.
 */
AsyncPool::GLThread AsyncPool::getContextThread(const fyusenet::GfxContextLink& ctx, int timeout) {
    using namespace std::chrono_literals;
    if (!ctx.context_) THROW_EXCEPTION_ARGS(GLException,"No valid context supplied");
    lock_.lock();
    if (goingDown_) return GLThread();
    if (!watchDog_) watchDog_ = new std::thread(sniff);
    teardownProtection_.fetch_add(1);        // we unlock in the meantime and this prevents a teardown while we are working on stuff
    int waittime = 0;
    for (auto it = glThreads_.begin(); it != glThreads_.end(); ++it) {
        entry & ent = (*it);
        if ((static_cast<GLThreadImpl *>(ent.thread))->context_ == ctx) {
            uint32_t expected = 0;
            while (!ent.refcount.compare_exchange_strong(expected, 1)) {
                lock_.unlock();
                // TODO (mw) polling, get rid of that
                std::this_thread::sleep_for(5ms);
                waittime += 5;
                if ((timeout >= 0) && (waittime >= timeout)) {
                    return GLThread();
                }
                lock_.lock();
                expected = 0;
            }
            teardownProtection_.fetch_sub(1);
            lock_.unlock();
            assert(ent.thread->isBusy() == false);
            return GLThread((GLThreadImpl *)ent.thread, &ent.refcount);
        }
    }
    if ((threads_.size() + glThreads_.size()) >= HARD_MAX_THREADS) {
        THROW_EXCEPTION_ARGS(GLException, "Maximum number of threads (%d) reached and trying to create one more, something is not right", HARD_MAX_THREADS);
    }
    GLThreadImpl * newthread = new GLThreadImpl(ctx);
    glThreads_.emplace_back(newthread,1);
    entry & ent = glThreads_.back();
    teardownProtection_.fetch_sub(1);
    lock_.unlock();
    return GLThread((GLThreadImpl *)ent.thread, &ent.refcount);
}


/**
 * @brief Remove all threads from asynchronous pool
 *
 * @note Must be called before cleaning up the GfxContextManager
 */
void AsyncPool::tearDown() {
    using namespace std::chrono_literals;
    std::lock_guard<std::mutex> lck(lock_);
    while (teardownProtection_.load() > 0) {
        std::this_thread::sleep_for(25ms);
    }
    goingDown_ = true;
    if (watchDog_) watchDog_->join();
    delete watchDog_;
    watchDog_ = nullptr;
    for (entry & ent : threads_) {
        while (ent.refcount.load() > 0) {
            std::this_thread::sleep_for(25ms);
        }
        ent.thread->wait();
        delete ent.thread;
    }
    threads_.clear();
    for (entry & ent : glThreads_) {
        while (ent.refcount.load() > 0) {
            std::this_thread::sleep_for(25ms);
        }
        ent.thread->wait();
        delete ent.thread;
    }
    glThreads_.clear();
    goingDown_ = false;
    watchDog_ = nullptr;
}


/**
 * @brief Check if pool has any GL-related threads
 *
 * @retval true if there is at least one GL-related thread
 * @retval false otherwise
 */
bool AsyncPool::isEmpty() {
    std::lock_guard<std::mutex> lck(lock_);
    return (glThreads_.size() == 0);
}


/**
 * @brief Log basic pool statistics for debugging
 */
void AsyncPool::logStatistics() {
    std::lock_guard<std::mutex> lck(lock_);
#ifdef DEBUG
    FNLOGD("AsyncPool Statistics:");
    FNLOGD("    # requests: %" PRIu64 "", requests_);
    FNLOGD("    # immediate hits: %" PRIu64 "", immediateHit_);
    FNLOGD("    # wait cycles: %" PRIu64 "", waitCycles_);
    FNLOGD("    wait time (ms): %" PRIu64 "", waitCycles_*2);
    FNLOGD("    reaped threads: %d", reapCount_);
#endif
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @brief Constructor
 *
 * Constructs a thread that is waiting for small tasks (tasklets if you want) to be executed by
 * this thread.
 */
AsyncPool::ThreadImpl::ThreadImpl() {
    using namespace std::chrono_literals;
    lastUsed_ = std::chrono::steady_clock::now();
    thread_ = new std::thread([this]() {
        std::unique_lock<std::mutex> locker(lock_);
        quit_ = false;
        do {
            waiting_.wait(locker,[this]() { return (task_); });
            if ((!quit_) && (task_)) {
                task_();
            }
            task_ = EMPTY;
            assert(!(task_));
            {
                std::unique_lock<std::mutex> readylck(taskDoneMutex_);
                taskDone_.notify_one();
            }
        } while (!quit_);
    });
    // wait for thread to come up
    while (quit_) {
        std::this_thread::sleep_for(25ms);
    }
}


/**
 * @brief Destructor
 *
 * Signals the thread to stop exeuction and waits blockingly until thread has joined.
 */
AsyncPool::ThreadImpl::~ThreadImpl() {
    std::unique_lock<std::mutex> il(issueLock_);
    do {
        lock_.lock();
        if (task_) lock_.unlock();
        else break;
    } while (!quit_);
    assert(!task_);
    task_ = [this] { quit_ = true; };
    waiting_.notify_one();
    lock_.unlock();
    thread_->join();
    delete thread_;
}

/**
 * @brief Constructor
 *
 * @param ctx Link to GL context that the thread should run with
 *
 * Constructs a thread that is waiting for small tasks (tasklets if you want) to be executed by
 * this thread. The thread itself is attached to the provided GL context, such that all tasks
 * sent to this thread are executed with that context being the current one.
 */
AsyncPool::GLThreadImpl::GLThreadImpl(const fyusenet::GfxContextLink& ctx) : ThreadImpl(), context_(ctx) {
    waitTask([this]() {
        bool rc = AsyncPool::makeCurrent(context_);
        assert(rc);
    });
}

/**
 * @brief Set a new task to the thread
 *
 * @param func Task to be executed
 *
 * @retval true if task was successfully placed
 * @retval false if task could not be placed because thread was quitting
 *
 * This function waits for the GLThread instance to become available for task execution and sets
 * a new task to be executed. It then sets the supplied \p func as new task and returns.
 */
bool AsyncPool::ThreadImpl::setTask(std::function<void()> func) {
    std::unique_lock<std::mutex> il(issueLock_);
    do {
        lock_.lock();
        if (task_) lock_.unlock();
        else break;
    } while (!quit_);
    bool r = true;
    if (!quit_) {
        assert(!(task_));
        task_ = func;
        waiting_.notify_one();
    } else r = false;
    lock_.unlock();
    return r;
}


/**
 * @brief Set a new task to the thread and wait for its completion
 *
 * @param func Task to be executed
 *
 * @retval true if task was successfully placed and executed
 * @retval false if task could not be placed because thread was quitting
 *
 * This function waits for the GLThread instance to become available for task execution and sets
 * a new task to be executed. It then sets the supplied \p func as new task and waits for the
 * execution of the task to be completed.
 */
bool AsyncPool::ThreadImpl::waitTask(std::function<void()> func) {
    std::unique_lock<std::mutex> il(issueLock_);
    do {
        lock_.lock();
        if (task_) lock_.unlock();
        else break;
    } while (!quit_);
    assert(!(task_));
    bool r = true;
    if (!quit_) {
        task_ = func;
        {
            std::unique_lock<std::mutex> locker(taskDoneMutex_);
            waiting_.notify_one();
            lock_.unlock();
            taskDone_.wait(locker,[this] { return !(task_);});
        }
    } else {
        r = false;
        lock_.unlock();
    }
    return r;
}


/**
 * @brief Wait for task on thread to be completed
 */
void AsyncPool::ThreadImpl::wait() {
    std::unique_lock<std::mutex> issue(issueLock_);
    std::unique_lock<std::mutex> locker(taskDoneMutex_);
    taskDone_.wait(locker,[this] { return !(task_);});
}


/**
 * @brief Check if thread is busy executing a task
 *
 * @retval true if thread is currently busy executing a task
 * @retval false otherwise
 */
bool AsyncPool::ThreadImpl::isBusy() const{
    std::unique_lock<std::mutex> il(issueLock_);
    return bool(task_);
}


/**
 * @brief Helper function to make the supplied context the current one to the thread
 *
 * @param ctx Link to GL context that should be made the current one to the calling thread
 *
 * @retval true on success
 * @retval otherwise
 *
 * @note On success, the context cannot be used in the original thread anymore without potentially
 *       causing trouble.
 */
bool AsyncPool::makeCurrent(const fyusenet::GfxContextLink& ctx) {
    if (ctx.context_) return ctx.context_->makeCurrent();
    else return false;
}


/**
 * @brief Watchdog function that checks whether threads have been idle and readping them
 *
 * This is a watchdog function that checks all threads for activity. Threads that have not
 * been active for a long time are reaped from the pool in order to conserve resources.
 */
void AsyncPool::sniff() {
    using namespace std::chrono_literals;
    using msec = std::chrono::milliseconds;
    std::vector<ThreadImpl *> reap;
    while (!goingDown_) {
        lock_.lock();
        teardownProtection_.fetch_add(1);
        auto now = std::chrono::steady_clock::now();
        auto it = threads_.begin();
        while (it != threads_.end()) {
            entry & ent = (*it);
            if ((ent.refcount.load() == 0) && (!ent.thread->isBusy())) {
                auto dura = std::chrono::duration_cast<msec>(now - ent.thread->lastUsed_);
                if (dura.count() > INACTIVITY_TIMEOUT) {
                    reap.push_back(ent.thread);
                    reapCount_++;
                    it = threads_.erase(it);
                } else ++it;
            } else ++it;
        }
        teardownProtection_.fetch_sub(1);
        lock_.unlock();
        for (auto * thread : reap) delete thread;
        reap.clear();
        std::this_thread::sleep_for(1000ms);
    }
}

} // opengl namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
