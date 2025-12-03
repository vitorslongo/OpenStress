import matplotlib.pyplot as plt

gl = [6, 20, 80, 160]
tensoes_maximas = [62.425, 120.734, 273.069, 427.588]

plt.plot(gl, tensoes_maximas)
plt.title("GL do sistema X Tensões máximas")
plt.xlabel("GL")
plt.ylabel("Tensões máximas")
plt.grid()
plt.show()