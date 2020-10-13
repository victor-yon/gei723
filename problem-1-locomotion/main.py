import logging

from HexaBob import HexaBob

if __name__ == '__main__':
    # Define the logging level
    logging.getLogger().setLevel(logging.INFO)

    # Create an new Bob!
    bob = HexaBob(nb_legs_pair=3,
                  sensor_front=0,
                  sensor_back=0.2,
                  sensor_left=0,
                  sensor_right=0)

    # Print Bob
    print(bob)

    # Run the network simulation during a specific time in ms, then plot some figures
    bob.run(duration=300)
