from flask import Flask, render_template, request, jsonify
import pickle
import uuid

app = Flask(__name__, template_folder='web/templates', static_folder='web/static')

# Cargar el modelo al iniciar
with open('modelo/modelo_entrenado.pkl', 'rb') as f:
    modelo = pickle.load(f)

@app.route('/')
def inicio():
    return render_template('index.html')

@app.route('/clasificar', methods=['POST'])
def clasificar():
    datos = request.get_json()
    descripcion = datos.get('descripcion', '')
    subject = datos.get('subject', '')
    
    texto_completo = subject + ' ' + descripcion
    
    # Predecir
    categoria = modelo.predecir(texto_completo)
    puntajes = modelo.predecir_con_probabilidades(texto_completo)
    
    # Generar ID de ticket
    ticket_id = str(uuid.uuid4())[:8].upper()
    
    return jsonify({
        'ticket_id': ticket_id,
        'categoria': categoria,
        'puntajes': puntajes
    })

if __name__ == '__main__':
    app.run(debug=True)