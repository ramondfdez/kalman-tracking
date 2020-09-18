import numpy as np 
from kalmanFilter import KalmanFilter
from scipy.optimize import linear_sum_assignment
from collections import deque


class Track(object):
	"""Track class for every object to be tracked"""
	
	def __init__(self, detection, trackId):
		self.trackId = trackId		
		self.KF = KalmanFilter()
		self.prediction = np.asarray(prediction)
		self.skipped_frames = 0  # number of frames skipped undetected
		self.trace = []  # trace path

class Tracker(object):
	"""Tracker class that updates track vectors of object tracked"""
	
	def __init__(self, dist_threshold, max_frame_skipped, max_trace_length, trackId):
		super(Tracker, self).__init__()
		self.dist_threshold = dist_threshold
		self.max_frame_skipped = max_frame_skipped
		self.max_trace_length = max_trace_length
		self.trackId = trackId
		self.tracks = []

	def update(self, detections):
		"""Update tracks vector using following steps:
		    - Create tracks if no tracks vector found
		    - Calculate cost using sum of square distance
		      between predicted vs detected centroids
		    - Using Hungarian Algorithm assign the correct
		      detected measurements to predicted tracks
		      https://en.wikipedia.org/wiki/Hungarian_algorithm
		    - Identify tracks with no assignment, if any
		    - If tracks are not detected for long time, remove them
		    - Now look for un_assigned detects
		    - Start new tracks
		    - Update KalmanFilter state, lastResults and tracks trace """

		
		# Create tracks if no tracks vector found
		if len(self.tracks) == 0:
			for i in range(len(detections)):
				track = Tracks(detections[i], self.trackId)
				self.trackId +=1
				self.tracks.append(track)
				
		# Calculate cost using sum of square distance between
		# predicted vs detected centroids
		N = len(self.tracks)
		M = len(detections)
		cost =  np.zeros(shape=(N, M))
		for i in range(N):
			for j in range(M):
				try:
					diff = self.tracks[i].prediction - detections[j]
					distance = np.sqrt(diff[0][0]*diff[0][0] + diff[1][0]*diff[1][0])
					cost[i][j] = distance
				except:
					pass

		cost = np.array(cost)*0.5
		
		# Using Hungarian Algorithm assign the correct detected measurements
		# to predicted tracks
		assignment = []
		for _ in range(N):
			assignment.append(-1)
			
		row, col = linear_sum_assignment(cost)
		for i in range(len(row)):
			assignment[row[i]] = col[i]
			
        	# Identify tracks with no assignment, if any
		un_assigned_tracks = []

		for i in range(len(assignment)):
			if (assignment[i] != -1):
				if (cost[i][assignment[i]] > self.dist_threshold):
					assignment[i] = -1
					un_assigned_tracks.append(i)
			else:
				self.tracks[i].skipped_frames += 1
				
		# If tracks are not detected for long time, remove them
		del_tracks = []
		for i in range(len(self.tracks)):
			if self.tracks[i].skipped_frames > self.max_frame_skipped :
				del_tracks.append(i)

		if len(del_tracks) > 0:
			for i in range(len(del_tracks)):
				if i < len(self.tracks):
					del self.tracks[i]
					del assignment[i]
				else:
					print("ERROR: id is greater than length of tracks")
				
		# Now look for un_assigned detects
		un_assigned_detects = []
		for i in range(len(detections)):
			if i not in assignment:
				un_assigned_detects.append(i)
				
		# Start new tracks
		if(len(un_assigned_detects) != 0):
			for i in range(len(un_assigned_detects)):
				track = Track(detections[un_assigned_detects[i]], self.trackId)
				self.trackId += 1
				self.tracks.append(track)
				

		# Update KalmanFilter state, lastResults and tracks trace
		for i in range(len(assignment)):
		    self.tracks[i].KF.predict()

		    if(assignment[i] != -1):
			self.tracks[i].skipped_frames = 0
			self.tracks[i].prediction = self.tracks[i].KF.correct(
						    detections[assignment[i]], 1)
		    else:
			self.tracks[i].prediction = self.tracks[i].KF.correct(
						    np.array([[0], [0]]), 0)

		    if(len(self.tracks[i].trace) > self.max_trace_length):
			for j in range(len(self.tracks[i].trace) -
				       self.max_trace_length):
			    del self.tracks[i].trace[j]

		    self.tracks[i].trace.append(self.tracks[i].prediction)
		    self.tracks[i].KF.lastResult = self.tracks[i].prediction








		



