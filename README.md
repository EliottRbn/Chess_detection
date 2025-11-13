# Chess_detection
This is the repository for our chess detection project in "Conception des systèmes - Prototypage rapide" at IPSA.


### Run the code
We are running an Expo App to fetch data from the server to display the pieces and the chessboard found

To run the server, run `server_v2.py `and send the images to format 640x640 with a POST methodK, HTTP.

The Expo App capture the camera stream, and send one photo every second. The server do the processing for each frame received, and send the bounding box for each piece found and the chessboard found.

### Test the code

The code can be run without the server and the Expo App. You just need to run `test_inference.py`. 
Change the path of the image to test with another image