import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby, count

Nx = int(2e+06) # Resolution parameter
setWidth = 0.2 # Width of square domain (return set)
delta = 0.25*setWidth # Can make more informed decision about this later

# Set up line 1, the leftmost line bounding the return set, parallel to the unstable direction.
line1 = [[setWidth*ii/Nx,setWidth] for ii in range(Nx+1)]
line1 = np.array(line1).T
## Rotate w.r.t unstable eigenvector
rotAngle = np.arctan(0.5*(np.sqrt(5)-1))
c = np.cos(rotAngle)
s = np.sin(rotAngle)
rotMat = np.array(((c,-s),(s,c))) # Anticlockwise
clockRotMat = np.array(((c,s),(-s,c))) # Clockwise
line1 = np.matmul(rotMat,line1)
## Translate to centre of domain
centroid = np.matmul(rotMat,np.array(((setWidth/2),(setWidth/2))))
centroid = centroid[:,np.newaxis]
translation = 0.5-centroid
line1 = line1 + translation

# Boundaries of square (for visualisation)
corners = setWidth*np.array(((0,0,1,1,0),(0,1,1,0,0)))
corners = np.matmul(rotMat,corners)
corners = corners + translation

# Plot return set boundary, initial line
plt.plot(corners[0,:],corners[1,:],'k')
plt.scatter(line1[0,:],line1[1,:],marker='.',c='b',s=2)
plt.xlim(0,1)
plt.ylim(0,1)
plt.savefig('start.png')
plt.show()
plt.close()



# Define the cat map
catMap = np.array(((2,1),(1,1)))

# Work out clockwise rotated return set (makes intersection analysis easier)
xlims = [translation[0],translation[0]+0.2]
ylims = [translation[1],translation[1]+0.2]
eigenval = (3-np.sqrt(5))/2 # Shrinking factor in stable eigenvector direction

listOfLines = [line1]

returnStats = []

for ii in range(10):
	areaReturned = 0
	newListOfLines = []
	for line in listOfLines: # Potential for parallelising here
		
		line = (np.matmul(catMap,line))%1	
		
		# Analysis of intersection with return set
		L = np.matmul(clockRotMat,(line-translation)) + translation
		lenL = L.shape[1]
		indexList = []
		for jj in range(lenL):
			if xlims[0]<L[0,jj]<xlims[1]:
				# Condition that parallel line is also in return set
				if ylims[0]+0.2*eigenval**(ii+1)<L[1,jj]<ylims[1]:
					indexList.append(jj)
		iList = [list(g) for k, g in groupby(indexList, key=lambda i,j=count(): i-next(j))]	
		# iList is the list of index sets corresponding to different lines in return set
		indexBoundaries = [0] # Will contain index boundaries of new lines 
		for subL in iList: # subL is the indices of the points of intersection of L with square domain
			# Check that delta condition has been met...
			kk = subL[-1]
			counter = 0
			while counter == 0 and kk < lenL-1: # upper delta condition hasn't yet been met and kk is still in the range of possible indices
				kk += 1
				if L[0,kk]>xlims[1]+delta:
					counter = 1 # Upper delta condition has been met
			kk = subL[0]
			while counter == 1 and 0 < kk: # as above, but for lower delta condition
				kk -= 1
				if L[0,kk]<xlims[0]-delta:
					counter = 2 # Lower delta condition has been met
					# Now we need to delete points intersecting return zone
					areaReturned += eigenval**(ii+1) # Normalised by area of return set
					indexBoundaries.append(subL[0])
					indexBoundaries.append(subL[-1]+1)
		indexBoundaries.append(lenL)
		if len(indexBoundaries) == 2:
			newListOfLines.append(line)
		else:
			for bb in range(int(0.5*len(indexBoundaries))):
				newListOfLines.append(line[:,indexBoundaries[2*bb]:indexBoundaries[2*bb+1]])
	listOfLines = newListOfLines
	returnStats.append(areaReturned)
	fd = open('/usr/not-backed-up/catMap/variableDelta/'+str(Nx)+'.csv','a')
	fd.write(str(ii+1)+','+str(areaReturned)+'\n')
	fd.close()
	
	plt.plot(corners[0,:],corners[1,:],'k')
	for line in listOfLines:
		plt.scatter(line[0,:],line[1,:],marker='.',c='b',s=2)
	plt.xlim(0,1)
	plt.ylim(0,1)
	plt.savefig(str(ii)+'.png')
	plt.show()
	plt.close()
	
