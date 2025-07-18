1. replay, working with chunks in and one-deque out, every iter given a source as
   {key: (batch_size,  context + batch_length * consec, len_bytes if key is 'stepid')}
   the first column for key 'is_first' is annotated as True even if not the 1st of episodes
2. stream, every next given a consec from the current source as
   {key: (batch_size,  context + batch_length, len_bytes if key is 'stepid')}
   set the elements for key 'consec' as the consec index, if used up source, iter replay
3. step, interact with the environments (there can be a batch of environments)
   driver.step, record transitions for all envs as 
   {updated (current) obs (with 'reward', 'is_first', 'is_last', 'log/', ... keys)
    policy determined (current) action, if obs is_last, this action will be reset
    outs, including the dynamical entries produced by the current world model
    log infomation, pop from current obs
   }
   the transitions are added to the replay buffer with the current dynamical entries
   each env uses a worker
   other callbacks, the increment of step and the step of fps need no tran and worker
   the logfn, record tran for episodes, for batch envs according to worker
   episode stats sent to logger with should_log
4. train, given a chunk from stream, apply context
   if the first chunk (indicated by the consec index), bootstrap recurrent states by context
   else the context seems to be ignored, maybe because the entries have just been updated
   the input entries are bootstraped if necessary by the pre-updated world model
   then the world model is updated, being used for the bootstrap in the following training
   train stats sent to logger with should_log
   agent internal stats sent to logger with should_report, less frequent than should_log
5. update, 
    DreamerV3 does not update all dynamical entries in the replay buffer explicitly after every world model update. Instead, it uses a context-based bootstrapping mechanism and only updates the replay buffer with new dynamical entries (encoder/decoder entries always empty) for the most recent batch, if and only if replay_context is enabled.

    in the replay.update method, it is possible for the same stepid to be updated multiple times by _setseq, depending on the structure of the input stepid array and the data being passed in.
6. should_report, should_log, should_save
   logger.write if should_log, can be traced by scope.viewer
   (frontend in src built with npm into dist, function Reversed may need to be fixed)
   (backend works with fastapi functions in server.py, 
    activated by frontend, process and send data to entry point,
    the data will refresh the frontend)
7. 3,4,5,6 loop