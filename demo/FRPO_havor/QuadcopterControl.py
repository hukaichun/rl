import numpy as np
import math
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as Axes3D

class QuadcopterControl(object):
	"""docstring for QuadcopterControl"""
	def __init__(self):

		# robot states
		self.q = np.zeros(7)	# Quaternion, Position
		self.u = np.zeros(6)	# Anguler velocity, Velocity
		self.du = np.zeros(6)	# Anguler acceleration, Acceleration
		self.R = np.zeros((3,3))

		# robot parameters
		self.length = 0.105;
		self.dragCoeff = 0.016;
		self.mass = 0.667;

		# set default parameters
		self.suicide_param = 0
		self.valueAtTermination = 1.5
		self.discountFactor = 0.99
		self.timeLimit = 15.0
		self.controlUpdate_dt = 0.01
		self.gravity = np.vstack([0.0, 0.0, 9.81])
		self.diagonalInertia = np.array([0.0023, 0.0025, 0.0037])
		self.inertia = np.mat(np.diag(self.diagonalInertia))
		self.inertiaInv = np.mat(self.inertia).I
		self.comLocation = np.array([0.0, 0.0, -0.05])

		# Scale
		self.actionScale = 6.0
		self.orientationScale = 1.0
		self.positionScale = 0.5
		self.angVelScale = 0.15
		self.linVelScale = 0.5

		# Adding constraints
		## X type
		self.transsThrust2GenForce = np.mat([[-self.length/math.sqrt(2), self.length/math.sqrt(2), self.length/math.sqrt(2), -self.length/math.sqrt(2)],
											 [self.length/math.sqrt(2), -self.length/math.sqrt(2), self.length/math.sqrt(2), -self.length/math.sqrt(2)],
											 [self.dragCoeff, self.dragCoeff, -self.dragCoeff, -self.dragCoeff],
											 [-1.0, -1.0, -1.0, -1.0]])

		## + type
		# self.transsThrust2GenForce = np.mat([[-self.length, self.length, 0, 0],
		# 									 [0, 0, self.length, -self.length],
		# 									 [self.dragCoeff, self.dragCoeff, -self.dragCoeff, -self.dragCoeff],
		# 									 [-1.0, -1.0, -1.0, -1.0]])
		self.transsThrust2GenForceInv = self.transsThrust2GenForce.I

		# Render
		self.init_plot()
		self.renderCounter = 0

	def reset(self):
		self.q = np.vstack(np.concatenate([self.randQ(), (np.random.ranf(size=3) - 0.5) * 4]))
		self.u = np.vstack(np.concatenate([(np.random.ranf(size=6) - 0.5) * 2]))

		# self.q = np.zeros((7,1))	# Quaternion, Position
		# self.q[0] = 1
		# self.u = np.zeros((6,1))	# Anguler velocity, Velocity
		state = self.getState()
		self.suicide_param += 1e-5
		return state

	def step(self, action_t):

		orientation = self.q[0:4]
		self.R = np.mat(self.quat2transform(orientation))

		w_B = self.u[0:3]

		# Control input from action
		action_cmd = np.clip((self.actionScale * action_t + 0.25 * self.mass * self.gravity[-1]), 1e-8, 8.0)
		actionGenForce = self.transsThrust2GenForce * np.vstack(action_cmd)
		B_torque = actionGenForce[0:3]
		B_force = np.vstack([0.0, 0.0, actionGenForce[3]])

		du_w = (self.inertiaInv * (B_torque - np.cross(w_B, (self.inertia * w_B), axis=0)))
		du_a = (self.R * B_force) / self.mass + self.gravity

		self.du = np.vstack([du_w, du_a])
		# self.du = np.vstack([du_w, np.zeros((2,1)), du_a[2,0]])

		# Intergrate states
		self.u += self.du * self.controlUpdate_dt
		w_B = self.u[0:3]
		self.q[0:4] = self.boxplusI_Frame(self.q[0:4], w_B)
		self.q[0:4] = self.normalize(self.q[0:4])
		self.q[-3::] = self.q[-3::] + self.u[-3::] * self.controlUpdate_dt
		# self.q = np.array([q_o[0], q_o[1], q_o[2], q_o[3], q_p[0], q_p[1], q_p[2]], dtype=float)

		# clip velocity???
		self.u[0:3] = np.clip(self.u[0:3], -20.0, 20.0)
		self.u[3:6] = np.clip(self.u[3:6], -5.0, 5.0)

		state = self.getState()
		# sensor_disturbance = np.concatenate([(np.random.ranf(size=9) - 0.5) * 0 *state[0:9],
		# 			 				  		 (np.random.ranf(size=6) - 0.5) * 0.5 *state[9:15],
		# 			 				  		 (np.random.ranf(size=3) - 0.5) * 0.5 *state[15:18]])
		# state += sensor_disturbance

		costOUT = 0.002 * np.linalg.norm(self.diffQ(np.array([1.0, 0.0, 0.0, 0.0]), np.hstack(self.q[0:4]))) \
				+ 0.002 * np.linalg.norm(self.q[4:7]) \
				+ 0.002 * np.linalg.norm(action_t)

		if np.linalg.norm(self.q[4:7]) < 0.005:
			done = True
			costOUT -= 100
		else:
			done = False
		# done = False

		return state, -costOUT, done, None


	def getState(self):
		orientation = self.q[0:4]
		self.q[0:4] = self.normalize(self.q[0:4])
		R = self.quat2transform(orientation)

		return np.concatenate([np.hstack(R),
							np.hstack(self.u) * self.angVelScale,
							np.hstack(self.q[4:7]) * self.positionScale])

	def normalize(self, v):
		norm=np.linalg.norm(v)
		if norm==0:
			norm=np.finfo(v.dtype).eps
		return v/norm

	def diffQ(self, q1, q2):
		diff = np.zeros(4)
		diff[0] = 1 - abs(q1[0] * q2[0] + np.dot(q1[1:4], q2[1:4]))
		diff[1:4] = q1[0] * q2[1:4] + q2[0] * q1[1:4] + np.cross(q1[1:4], q2[1:4])
		return diff

	def quat2transform(self, quaternion):
		"""
		Transform a unit quaternion into its corresponding rotation matrix (to
		be applied on the right side).

		:returns: transform matrix
		:rtype: numpy array

		"""
		w, x, y, z = quaternion
		xx2 = 2 * x * x
		yy2 = 2 * y * y
		zz2 = 2 * z * z
		xy2 = 2 * x * y
		wz2 = 2 * w * z
		zx2 = 2 * z * x
		wy2 = 2 * w * y
		yz2 = 2 * y * z
		wx2 = 2 * w * x

		rmat = np.empty((3, 3), float)
		rmat[0,0] = 1. - yy2 - zz2
		rmat[0,1] = xy2 - wz2
		rmat[0,2] = zx2 + wy2
		rmat[1,0] = xy2 + wz2
		rmat[1,1] = 1. - xx2 - zz2
		rmat[1,2] = yz2 - wx2
		rmat[2,0] = zx2 - wy2
		rmat[2,1] = yz2 + wx2
		rmat[2,2] = 1. - xx2 - yy2

		return rmat

	def randQ(self):
		quaternion = self.normalize((np.random.ranf(size=4) - 0.5) * 2)

		return quaternion

	def boxplusI_Frame(self, quaternion, angulerVelocity):
		p, q, r = np.hstack(angulerVelocity)
		dq = 0.5 * np.mat([[0, -p, -q, -r],
					[p, 0, r, -q],
					[q, -r, 0, p],
					[r, q, -p, 0]]) * np.vstack(quaternion)
		return quaternion + dq * self.controlUpdate_dt

	def init_plot(self):
		fig = plt.figure()
		ax = Axes3D.Axes3D(fig)
		ax.set_xlim3d([-1.0, 1.0])
		ax.set_xlabel('X (m)')
		ax.set_ylim3d([-1.0, 1.0])
		ax.set_ylabel('Y (m)')
		ax.set_zlim3d([1.0, -1.0])
		ax.set_zlabel('Z (m)')
		ax.set_title('Quadcopter Hover Simulation')

		self.l1, = ax.plot([],[],[],color='blue',linewidth=1,antialiased=False)
		self.l2, = ax.plot([],[],[],color='red',linewidth=1,antialiased=False)
		self.l3, = ax.plot([],[],[],color='green',linewidth=1,antialiased=False)
		self.hub, = ax.plot([],[],[],marker='o',color='green', markersize=3,antialiased=False)
		self.center, = ax.plot([],[],[],marker='o',color='red', markersize=3,antialiased=False)
		# plt.show()

	def render(self):

		if self.renderCounter % 10 == 0:
			points = np.array([[-0.074,0.074,0], [0.074,-0.074,0], [0.074,0.074,0], [-0.074,-0.074,0], [0,0,-0.03], [0,0,0] ]).T
			points = np.dot(self.quat2transform(self.q[0:4]), points).T
			points += self.q[4:7].T
			center_point = np.array([0,0,0])
			self.l1.set_data(points[0:2,0],points[0:2,1])
			self.l1.set_3d_properties(points[0:2,2])
			self.l2.set_data(points[2:4,0],points[2:4,1])
			self.l2.set_3d_properties(points[2:4,2])
			self.l3.set_data(points[4:6,0],points[4:6,1])
			self.l3.set_3d_properties(points[4:6,2])
			self.hub.set_data(points[5,0],points[5,1])
			self.hub.set_3d_properties(points[5,2])
			self.center.set_data(center_point[0],center_point[1])
			self.center.set_3d_properties(center_point[2])
			plt.pause(0.0000000000000000001)

			self.renderCounter = 1

		else:
			self.renderCounter += 1

if __name__ == "__main__":
	a = QuadcopterControl()
	for i in range(1,10):
		a.reset()
		for x in range(1,200):
			state, reward, done, _ = a.step(np.array([0.,0.,0.,0.]))
			a.render()
			# print (reward)
