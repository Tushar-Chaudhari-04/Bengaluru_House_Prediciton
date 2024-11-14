from app import app,load_save_artifacts

if __name__ == '__main__':
    load_save_artifacts()
    print("loading prediction...")
    app.run(debug=True)