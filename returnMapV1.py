import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby, count
import collections

setWidth = 0.2 # Width of square domain (return set)
delta = 0.25*setWidth # Can make more informed decision about this later

'''
# Set up line 1, the leftmost line bounding the return set, parallel to the unstable direction.
line1 = [[setWidth*ii/Nx,setWidth] for ii in range(Nx+1)]
line1 = np.array(line1).T
'''
## Rotate w.r.t unstable eigenvector
rotAngle = np.arctan(0.5*(np.sqrt(5)-1))
c = np.cos(rotAngle)
s = np.sin(rotAngle)
rotMat = np.array(((c,-s),(s,c))) # Anticlockwise
clockRotMat = np.array(((c,s),(-s,c))) # Clockwise

## Translate to centre of domain
centroid = np.matmul(rotMat,np.array(((setWidth/2),(setWidth/2))))
centroid = centroid[:,np.newaxis]
translation = 0.5-centroid

# Boundaries of square (for visualisation)
corners = setWidth*np.array(((0,0,1,1,0),(0,1,1,0,0)))
corners = np.matmul(rotMat,corners)
corners = corners + translation

# Lines from x0->x1 are set up as 2x2 matrices (x0 x1
#				    		y0 y1 ) can be easily mapped forward using matrix mult.
initialLine = corners[:,1:3]


deltaPoints = np.array(((-delta,setWidth+delta),(0,0)))
deltaPoints = np.matmul(rotMat,deltaPoints)
deltaPoints = deltaPoints + translation

# Delta condition line info: y = mx + c Parallel to STABLE direction
ms = (corners[1,0]-corners[1,1])/(corners[0,0]-corners[0,1])
cDeltaUpper = deltaPoints[1,1]-ms*deltaPoints[0,1]  # upper delta condition line
cDeltaLower = deltaPoints[1,0]-ms*deltaPoints[0,0]  # upper delta condition line

# Unstable gradient
mu = (corners[1,2]-corners[1,1])/(corners[0,2]-corners[0,1])
# Intersect return set condition, crossings with 'y axis'
cUpper = corners[1,1]-mu*corners[0,1]
cLower = corners[1,0]-mu*corners[0,0]

# As above, but for the stable direction
cStableUpper = corners[1,2]-ms*corners[0,2]
cStableLower = corners[1,1]-ms*corners[0,1]


print(cStableUpper, cStableLower)


'''
plt.plot(initialLine[0,:],initialLine[1,:],'k')
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()
plt.close()
'''

'''
# Plot return set boundary, initial line
plt.plot(corners[0,:],corners[1,:],'k')
plt.xlim(0,1)
plt.ylim(0,1)
plt.savefig('start.png')
plt.show()
plt.close()
'''

# Define the cat map
catMap = np.array(((2,1),(1,1)))

eigenval = (3-np.sqrt(5))/2 # Shrinking factor in stable eigenvector direction

listOfLines = [initialLine]

returnStats = []

for ii in range(500):
	print('Iteration '+str(ii+1)+', considering '+str(len(listOfLines))+' lines')
	print('Cat-Mapping, finding new lines')
	areaReturned = 0
	newListOfLines = []
	for line in listOfLines: # Potential for parallelising here
		# Work out new lines created, add to new list of lines
		line = (np.matmul(catMap,line)) # Gives a line in [0,3]x[0,2]
		xInt = []
		yInt = []
		if np.floor(line[0,1])-np.floor(line[0,0])>0:
			xInt = list(set([np.floor(line[0,1]),np.ceil(line[0,0])]))
		if len(xInt)>0:
			yInt += [line[1,0] + mu*(xx-line[0,0]) for xx in xInt]
		if np.floor(line[1,1])-np.floor(line[1,0])>0:
			xInt.append(line[0,0] + (1-line[1,0])/mu)
			yInt.append(1)
		xInt.sort()
		yInt.sort()
		
		xList = np.array([line[0,0]]+ xInt + [line[0,1]])
		yList = np.array([line[1,0]]+ yInt + [line[1,1]])
		for ll in range(len(xList)-1):
			nL = np.array([[xList[ll],xList[ll+1]],[yList[ll],yList[ll+1]]])%1
			for aa in range(2):
				if nL[aa,1] == 0:
					nL[aa,1] = 1
			nLc = nL[1,0]-mu*(nL[0,0]) # Intersection of line with the 'y-axis'
			newListOfLines.append((nLc,nL))
	listOfLines = newListOfLines	


	print('Joining up lines')
	# Now join up the connecting lines based on repeats of coords and intersection with 'y-axis'
	newListOfLines = []
	listOfLines = sorted(listOfLines, key=lambda x: x[0]) # Sort by intersection with 'y-axis'
	cval = -10 # Nonsense initial cval
	for intLine in listOfLines:
		if abs(intLine[0]-cval)>1e-09: # Doesn't match previous c value
			newListOfLines.append([intLine[1]]) # Create new sublist
		else:
			newListOfLines[-1].append(intLine[1]) # Add to previous list otherwise
		cval = intLine[0]
	# We now have a list of lists of lines which share the same y-axis intersection
	listOfLines = []
	for subList in newListOfLines:
		xcoordList = [nL[0,0] for nL in subList] + [nL[0,1] for nL in subList] # List of all x-coords
		ycoordList = [nL[1,0] for nL in subList] + [nL[1,1] for nL in subList] # List of all y-coords

		xTester = [round(elem,9) for elem in xcoordList]
		
		if len(set(xTester)) == 0.5*(len(xTester) + 2): # Every number repeats except inital and end points
			listOfLines.append(np.array([[np.min(xcoordList),np.max(xcoordList)],[np.min(ycoordList),np.max(ycoordList)]]))
			# The above gives the new line which has no missing segments
		else:
			seen = set()
			doubleSeenIndices = []
			for kk in range(len(xcoordList)):
				P = round(xcoordList[kk],6)
				if P not in seen:
					seen.add(P)
				else:
					doubleSeenIndices.append(kk)
			xdoubleSeen = [xcoordList[pp] for pp in doubleSeenIndices]
			ydoubleSeen = [ycoordList[pp] for pp in doubleSeenIndices]
			xEnds = list(set(xcoordList)-set(xdoubleSeen))
			xEnds.sort()
			yEnds = list(set(ycoordList)-set(ydoubleSeen))
			yEnds.sort()
			for jj in range(int(0.5*len(xEnds))):
				try:
					nnL = np.array([[xEnds[2*jj],xEnds[2*jj+1]],[yEnds[2*jj],yEnds[2*jj+1]]])
					listOfLines.append(nnL)
				except:
					print(xEnds,yEnds)
					continue
			# This adds in the other, disconnected lines
	

	print('Checking for returns')
	# Now assess lines for potential return, log return areas and split lines
	newListOfLines = []
	areaReturned = 0
	for line in listOfLines:
		crossing = 0
		if mu*line[0,0] + cLower + (0.2/s)*eigenval**(ii+1) < line[1,0] < mu*line[0,0] + cUpper: # General intercept condition
			if line[1,0] < ms*line[0,0] + cDeltaLower: # Lower delta condition
				if line[1,1] > ms*line[0,1] + cDeltaUpper: # Upper delta condition
					# Crossing condition has now been met, now need to record area returned and split line.	
					crossing = 1
					areaReturned += eigenval**(ii+1)
					cLine = line[1,0] - mu*line[0,0]
					xLower = (cStableLower-cLine)/(mu-ms)
					xUpper = (cStableUpper-cLine)/(mu-ms)
					yLower = mu*xLower + cLine
					yUpper = mu*xUpper + cLine
					newListOfLines.append(np.array([[line[0,0],xLower],[line[1,0],yLower]]))
					newListOfLines.append(np.array([[xUpper,line[0,1]],[yUpper,line[1,1]]]))

		if crossing == 0:
			newListOfLines.append(line)
	listOfLines = newListOfLines

	fd = open('/usr/not-backed-up/catMap/newMethodResults.csv','a')
	fd.write(str(ii+1)+','+str(areaReturned)+'\n')
	fd.close()


	'''
	for line in listOfLines:
		plt.plot(line[0,:],line[1,:])
	plt.xlim(-0.1,1.1)
	plt.ylim(-0.1,1.1)
	plt.show()
	plt.close()
	'''	





'''


	
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
	# Now improve resolution of line	
	toBeDeleted = []
	for kk in range(len(listOfLines)):
		line = listOfLines[kk]
		listOfArrays=[]
		try:
			x0 = line[:,0] # Inital (x,y) coord of line
			counter = 0
			for jj in range(line.shape[1]-1):
				counter += 1
				if line[0,jj+1]<line[0,jj] or line[1,jj+1]<line[1,jj]:
					x1 = line[:,jj]
					xsp = np.linspace(x0[0],x1[0],num=2*counter)
					ysp = np.linspace(x0[1],x1[1],num=2*counter)
					newArray = np.vstack((xsp,ysp))
					listOfArrays.append(newArray)
					counter = 0
					x0 = line[:,jj+1]
			xsp = np.linspace(x0[0],line[0,-1],num=2*counter)
			ysp = np.linspace(x0[1],line[1,-1],num=2*counter)
			newArray = np.vstack((xsp,ysp))
			listOfArrays.append(newArray)

			listOfArrays = tuple(listOfArrays)
			newLine = np.concatenate(listOfArrays, axis = 1)
			listOfLines[kk] = newLine
		except:
			toBeDeleted.append(line)
			continue
	for item in toBeDeleted:
		listOfLines.remove(item)
	returnStats.append(areaReturned)
	
	fd = open('/usr/not-backed-up/catMap/'+str(Nx)+'.csv','a')
	fd.write(str(ii+1)+','+str(areaReturned)+'\n')
	fd.close()
	
	plt.plot(corners[0,:],corners[1,:],'k')
	for line in listOfLines:
		plt.scatter(line[0,:],line[1,:],marker='.',c='b',s=2)
	plt.xlim(0,1)
	plt.ylim(0,1)
	plt.savefig(str(ii)+'.png')
	plt.close()
'''
