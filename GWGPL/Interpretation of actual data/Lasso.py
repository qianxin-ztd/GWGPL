import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Times New Roman']
img = plt.imread('toy example.jpg')
plt.xlabel('longitude (°E)')
plt.ylabel('latitude (°N)')
plt.xticks([0,500,1000,1500],[110, 114.1, 118.2, 122.5])
plt.yticks([0,500,1000,1500,2000,2500],[42.8, 39.1, 35.4, 31.7, 28, 24.3])
plt.imshow(img)
plt.savefig('toy.jpg', dpi=600)