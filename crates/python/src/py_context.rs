// Scope tracking uses per-task storage (tokio::task_local!) with a thread-local
// fallback for synchronous callers (Python GIL thread, sync tests). The Python
// async execute functions snapshot the calling thread's scope top before entering
// the tokio runtime so that the parent scope propagates into the spawned task.
