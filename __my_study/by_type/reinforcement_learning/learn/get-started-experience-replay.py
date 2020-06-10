def experience_replay(self):
    if len(self.memory) < BATCH_SIZE:
        return
    batch = random.sample(self.memory, BATCH_SIZE)
    for state, action, reward, state_next, terminal in batch:
        q_update = reward
        if not terminal:
            q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
        q_values = self.model.predict(state)
        q_values[0][action] = q_update
        self.model.fit(state, q_values, verbose=0)