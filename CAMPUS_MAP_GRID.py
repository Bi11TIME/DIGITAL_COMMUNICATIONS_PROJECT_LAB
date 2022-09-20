import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np

def findDistance(BeaconI, BeaconF):
    d = np.sqrt(((BeaconF[0] - BeaconI[0]) ** 2) + ((BeaconF[1] - BeaconI[1]) ** 2))
    return d
def findLocation(D1, D2, D3):
    A = np.array([[2*(Beacon2[0]-Beacon1[0]), 2*(Beacon2[1]-Beacon1[1])], [2*(Beacon3[0]-Beacon2[0]), 2*(Beacon3[1]-Beacon2[1])]])
    B = np.array([(-1 * ((Beacon1[0] ** 2) - (Beacon2[0] ** 2) + (Beacon1[1] ** 2) - (Beacon2[1] ** 2) - (D1 ** 2) + (D2 ** 2))), (-1 * ((Beacon2[0] ** 2) - (Beacon3[0] ** 2) + (Beacon2[1] ** 2) - (Beacon3[1] ** 2) - (D2 ** 2) + (D3 ** 2)))])
    location = np.linalg.inv(A).dot(B)
    return location

url = r"C:\Users\bblac\OneDrive\Documents\TTUFALL22\Digital_Communications_Project_Lab\GUI\Campus.png"
image = plt.imread(url)
fig = plt.figure()
ax = fig.subplots()
a = ax.imshow(image, aspect='auto')


Beacon1 = [336,565]
Beacon2 = [795,484]
Beacon3 = [643,304]
Base = [746, 287]

#update location
Ulocation = [550, 150]

D1 = findDistance(Beacon1, Ulocation)
D2 = findDistance(Beacon2, Ulocation)
D3 = findDistance(Beacon3, Ulocation)
location = findLocation(D1, D2, D3)



plt.hlines(y=np.arange(0, 10)+0.5, xmin=np.full(10, 0)-0.5, xmax=np.full(10, 10)-0.5, color="black")
plt.grid(b=True,color='green')
plt.xticks(np.arange(0,950,25), rotation = 270)
plt.yticks(np.arange(0,750,25))
plt.scatter(Beacon1[0], Beacon1[1], color = 'Yellow')
plt.scatter(Beacon2[0], Beacon2[1], color = 'Yellow')
plt.scatter(Beacon3[0], Beacon3[1], color = 'Yellow')
plt.scatter(Base[0], Base[1], color = 'black')
plt.scatter(location[0], location[1], color = 'Red')
plt.text(location[0] + 10,  location[1], "1", color = 'Black', size = 15)
#plt.scatter(Location1[0], Location1[1], color = 'Red')
#plt.scatter(Location2[0], Location2[1], color = 'Red')
#plt.scatter(Location3[0], Location3[1], color = 'Red')
#plt.text(Location1[0] + 20,Location1[1] + 20,"1", color = 'White', size = 15)
#plt.text(Location2[0] + 20,Location2[1] + 20,"2", color = 'White', size = 15)
#plt.text(Location3[0] + 20,Location3[1] + 20,"3", color = 'White', size = 15)




plt.show()
