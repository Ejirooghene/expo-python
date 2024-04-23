from app import app
 
@app.route('/recommendation', methods=['GET'])
def recommmend():
    return 'hello'
