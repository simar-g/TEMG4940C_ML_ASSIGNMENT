from flask import Flask, render_template
import matplotlib.pyplot as plt
import io
import base64

# create a Flask app
app = Flask(__name__)

# define a route for the dashboard
@app.route('/')
def dashboard():
    # generate a plot using Matplotlib
    x = [1, 2, 3, 4, 5]
    y = [1, 4, 9, 16, 25]
    plt.plot(x, y)
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title('My Plot')

    # save the plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # encode the plot as a base64 string
    plot_data = base64.b64encode(buf.getvalue()).decode()

    # render the dashboard template with the plot data
    return render_template('dashboard.html', plot_data=plot_data)

# start the Flask app
if __name__ == '__main__':
    app.run()