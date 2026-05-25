import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution()

np.random.seed(42)
tf.compat.v1.set_random_seed(42)

n_samples = 1000
batch_size = 100
num_steps = 20000

X_data = np.random.uniform(0, 1, (n_samples, 1)).astype(np.float32)
y_data = 2 * X_data + 1 + np.random.normal(0, 0.2, (n_samples, 1)).astype(np.float32)

X = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, 1))
y = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, 1))

with tf.compat.v1.variable_scope("linear_regression"):
    k = tf.Variable(tf.random.normal((1, 1), stddev=0.1), name="slope")
    b = tf.Variable(tf.zeros((1,)), name="bias")

y_pred = tf.matmul(X, k) + b

loss = tf.reduce_mean(tf.square(y - y_pred))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(
    learning_rate=0.05
).minimize(loss)

display_step = 1000

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(num_steps):
        indices = np.random.choice(n_samples, batch_size, replace=False)
        X_batch = X_data[indices]
        y_batch = y_data[indices]

        _, loss_val, k_val, b_val = sess.run(
            [optimizer, loss, k, b],
            feed_dict={X: X_batch, y: y_batch}
        )

        if (step + 1) % display_step == 0:
            print(
                f"Епоха {step + 1}: loss = {loss_val:.6f}, "
                f"k = {k_val[0][0]:.4f}, b = {b_val[0]:.4f}"
            )

    final_k, final_b = sess.run([k, b])

print("\nФінальні параметри:")
print(f"k = {final_k[0][0]:.4f}")
print(f"b = {final_b[0]:.4f}")

X_line = np.linspace(0, 1, 100).reshape(-1, 1)
y_line = final_k[0][0] * X_line + final_b[0]

plt.scatter(X_data, y_data, s=10, label="Навчальні дані")
plt.plot(X_line, y_line, color="red", linewidth=3, label="Навчена пряма")
plt.xlabel("X")
plt.ylabel("y")
plt.title("TensorFlow: навчання лінійної регресії")
plt.legend()
plt.grid(True)
plt.show()