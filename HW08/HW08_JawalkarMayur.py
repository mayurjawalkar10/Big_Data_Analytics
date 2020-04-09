"""
Author: Mayur Sunil Jawalkar (mj8628)
        Kunjan Suresh Mhaske (km1556)
Big Data Analytics: Homework-08
Description: In this assignment we are implementing a gradient decent algorithm
             to fit a line closest to the given points.
"""

# Import Statements
import math  # TO compute mathematical operations
from matplotlib import pyplot as plt  # To plot the graph

# Initialize the global parameters.
INITIAL_THETA = 15  # Angle between the x axis and the line representing the shortest distance from the given line.
INITIAL_RHO = 9  # Shortest distance of the line from the center
INITIAL_ALPHA = 5  # Parameter to decide the step size
MINIMUM_ALPHA_DELTA = 0.025  # Hyper-parameter to decide the breaking condition
LEARNING_RATE = 0.995  # Hyper-parameter to decide the change in alpha

# List to store average distance from all points per iteration
DIST_LIST = []


def gradient_descent():
    """
    This function achieves the behavior of the gradient decent. It starts with initial values of theta, rho, and
    alpha. It continues ot tune the parameters in order to reduce the average distance between all the points and the line
    generated using theta and rho.
    """

    # Access the global variables
    globals()

    # Input data  points
    data_points = [[2, 10], [3, 9], [4, 8], [5, 7], [6, 6], [7, 5], [8, 4], [9, 3], [10, 2]]

    # Validate the correct input for learning rate. i.e., [0 <= learning_rate <= 1]
    if LEARNING_RATE >= 1.0:
        print("Learning Rate cannot be greater than or equal to 1.0")
    elif LEARNING_RATE <= 0:
        print("Learning Rate cannot be less than or equal to zero")

    # Initialize the variables using the input parameters
    theta = INITIAL_THETA
    rho = INITIAL_RHO
    alpha = INITIAL_ALPHA

    # Keep iterating until exit criteria is met
    while True:
        # Compute the average distance of all points from the line
        dist_step_2 = dst_from_pts_to_line(data_points, theta, rho)

        # Insert the initial distance in the list containing all distances.
        if len(DIST_LIST) == 0:
            DIST_LIST.append(dist_step_2)

        # Calculate the average distance of all points from the line generated using (theta, rho-alpha)
        dist_step_3a_minus = dst_from_pts_to_line(data_points, theta, rho-alpha)
        # Calculate the average distance of all points from the line generated using (theta, rho+alpha)
        dist_step_3a_plus = dst_from_pts_to_line(data_points, theta, rho+alpha)

        # Update the best distance and parameters which give the best results.
        if dist_step_3a_minus < dist_step_2:  # check if line drawn using (theta, rho-alpha) gives shortest distance.
            step_for_rho = -alpha  # decide the step for rho
            dst_best_yet = dist_step_3a_minus  # note the best distance so far
            rho += step_for_rho  # update the rho
        elif dist_step_3a_plus < dist_step_2:  # check if line drawn using (theta, rho+alpha) gives shortest distance.
            step_for_rho = alpha  # decide the step for rho
            dst_best_yet = dist_step_3a_plus  # note the best distance so far
            rho += step_for_rho  # update the rho
        else:  # default
            dst_best_yet = dist_step_2  # note best distance
            step_for_rho = alpha  # Note step for rho

        # repeat until best rho value is found.
        while True:
            # Compute the average distance from the line using theta and rho+step_for_rho
            dist_step_3a = dst_from_pts_to_line(data_points, theta, rho+step_for_rho)

            # Check if the new distance is better/shorter than the previous
            if dist_step_3a < dst_best_yet:
                rho += step_for_rho  # Update the rho
                dst_best_yet = dist_step_3a  # Update the best distance
            else:
                break

        # Compute the average distance of all points from the line
        dist_step_3b = dst_from_pts_to_line(data_points, theta, rho)
        # Calculate the average distance of all points from the line generated using (theta-alpha, rho)
        dist_step_3b_minus = dst_from_pts_to_line(data_points, theta-alpha, rho)
        # Calculate the average distance of all points from the line generated using (theta+alpha, rho)
        dist_step_3b_plus = dst_from_pts_to_line(data_points, theta+alpha, rho)

        # check for the shortest distance
        if dist_step_3b_minus < dist_step_3b:  # check if line drawn using (theta-alpha, rho) gives shortest distance.
            step_for_theta = -alpha  # Note -alpha as a step size theta
            dst_best_yet = dist_step_3b_minus  # note the best distance yet
            theta += step_for_theta  # Update the theta
        elif dist_step_3a_plus < dist_step_3b:  # check if line drawn using (theta-alpha, rho) gives shortest distance.
            step_for_theta = alpha  # Note alpha as a step size for theta
            dst_best_yet = dist_step_3b_plus  # Note the best distance yet
            theta += step_for_theta  # Update the theta
        else:  # default
            dst_best_yet = dist_step_3b  # Note best distance
            step_for_theta = alpha  # Note step for theta as alpha

        # Repeat until best value for theta is found.
        while True:
            # Calculate the average distance of all points from the line generated using (theta+step_for_theta, rho)
            dist_step_3b = dst_from_pts_to_line(data_points, theta+step_for_theta, rho)

            # Check if the new distance is better/shorter than the previous
            if dist_step_3b < dst_best_yet:
                theta += step_for_theta  # Update the theta
                dst_best_yet = dist_step_3b  # Update the best distance

                # Ensure that theta value doesn't exceed 180 and remain in range [-180, +180]
                if theta > 180:
                    theta -= 360

                # Ensure that theta value doesn't fall below -180 and remain in range [-180, +180]
                if theta < -180:
                    theta += 360
            else:
                break

        # Ensure that the rho value is positive as it is a shortest distance from the center and the line.
        # And it is better to have a positive value of distance. Update the value of theta to make the rho positive.
        if rho < 0:
            # Update the value of theta ensuring that theta remains in range [-180, +180]
            if theta <= 0:
                theta += 180
            elif theta > 0:
                theta -= 180
            # Make the rho positive
            rho = -rho

        # Calculate the distance of all points from the line generated using theta and rho
        # after fine tuning the parameters.
        dist_step_4 = dst_from_pts_to_line(data_points, theta, rho)

        # Insert the average distance to the list
        DIST_LIST.append(dist_step_4)

        # Print all the debug statements after each iteration
        print("Rho = {:9.5f} Theta = {:+9.6f} Alpha = {:+6.5f} Avg Dist = {:8.7f}"
              "".format(rho, theta, alpha, dist_step_4))

        # Print the debug statement to know when the distance is getting worse.
        if dist_step_2 < dist_step_4:
            print("Distance is getting worse, not better.")

        # Upate the value of alpha. It will make the fine tuning more granular and produce more accurate results.
        alpha = LEARNING_RATE*alpha

        # Check for the breaking condition
        if alpha > MINIMUM_ALPHA_DELTA:
            continue
        else:  # break if the alpha goes below the minimum value which alpha can take.
            break

    # Calculate the final value for the rho. i.e., the shortest distance of the line from the center
    dst_to_origin = dst_from_pts_to_line([[0, 0]], theta, rho)

    # Plot the Average distances of all the points per iteration
    plt.plot(DIST_LIST)
    plt.title("Average distance of all points from the line per iteration")
    plt.ylabel("Avg Distance from all points")
    plt.xlabel("Iterations")
    plt.show()

    # Print the statement to know about the total steps taken by the algorithm to converge
    print(f"\nThe gradient Decent is converged after {len(DIST_LIST)} iterations")

    # Print the shortest distance of all the points from the final line.
    print("The final average distance of all points from the line is {:7.5f}\n".format(DIST_LIST[-1]))

    # Print final results.
    print("A                    = {:+7.5f}".format(math.cos(math.radians(theta))))
    print("B                    = {:+7.5f}".format(math.sin(math.radians(theta))))
    print("Theta                = {:+7.5f} degrees".format(theta))
    print("Rho                  = {:+7.5f}".format(rho))
    print("Distance from Origin = {:+7.5f}".format(dst_to_origin))


def dst_from_pts_to_line(data_points, theta, rho):
    """
    This function computes the average distance of all the points specified in the data_points from the line generated
    using theta and rho.
    :param data_points: list of input points [[x1, y1],[x2,y2],...]
    :param theta: angle between the x axis and the line representing shortest distance to the imaginary line and center.
    :param rho: Shortest distance of the imaginary line to the center.
    """
    # Note the total number of points
    num_points = len(data_points)
    # Calculate the Coefficient A for our line
    A = math.cos(math.radians(theta))
    # Calculate the Coefficient B for our line
    B = math.sin(math.radians(theta))
    # consider -rho as a 3rd coefficient for our line.
    C = -rho

    # Compute the total distance of all the points from the line.
    ttl_dist = sum([abs(A*data_points[idx][0] + B*data_points[idx][1] + C) for idx in range(len(data_points))])
    # compute the average distance.
    avg_dist = ttl_dist/num_points
    return avg_dist


if __name__ == '__main__':
    """
    Execute only as a script.
    """
    gradient_descent()
