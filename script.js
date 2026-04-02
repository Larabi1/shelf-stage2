// ==================== CONFIGURATION ====================
const IMG_SIZE = 224;
const STABILIZATION_SECONDS = 5;
const STREAM_INTERVAL_MS = 1500;

let session = null;
let mapping = {};
let skuMap = {};
let isModelReady = false;

// États
let streamActive = false;
let streamTimer = null;
let streamStabilizationTimer = null;
let lastStableSku = null;
let streamVideo = null;
let streamTrack = null;

let inventoryActive = false;
let inventoryTimer = null;
let inventoryStabilizationTimer = null;
let inventoryLastStableSku = null;
let currentInventoryPrediction = null;
let inventoryVideo = null;
let inventoryTrack = null;
let inventoryList = [];
let inventoryCount = {};

// ==================== CHARGEMENT MODÈLE ONNX ====================
async function loadModel() {
    try {
        showLoading(true);
        
        console.log('1️⃣ Chargement mapping...');
        const mappingRes = await fetch('assets/idx_to_class.json');
        mapping = await mappingRes.json();
        console.log('✅ Mapping chargé:', Object.keys(mapping).length, 'classes');
        
        console.log('2️⃣ Chargement CSV...');
        const csvRes = await fetch('assets/sku_catalog.csv');
        const csvText = await csvRes.text();
        
        Papa.parse(csvText, {
            header: true,
            skipEmptyLines: true,
            complete: (result) => {
                result.data.forEach(row => {
                    if (row.sku_id) {
                        skuMap[row.sku_id] = {
                            category: row.category,
                            brand: row.brand,
                            product_name: row.product_name,
                            capacity: row.capacity,
                            emballage: row.emballage,
                            saveur: row.saveur
                        };
                    }
                });
                console.log('✅ CSV chargé:', Object.keys(skuMap).length, 'produits');
            }
        });
        
        console.log('3️⃣ Chargement modèle ONNX...');
        session = await ort.InferenceSession.create('assets/best-v4.onnx');
        
        isModelReady = true;
        showLoading(false);
        console.log('✅ Modèle ONNX chargé avec succès !');
        
        // Afficher un message de succès
        const statusDiv = document.createElement('div');
        statusDiv.style.cssText = 'position:fixed; bottom:20px; right:20px; background:#10b981; color:white; padding:10px 20px; border-radius:40px; z-index:9999;';
        statusDiv.innerHTML = '✅ Modèle prêt !';
        document.body.appendChild(statusDiv);
        setTimeout(() => statusDiv.remove(), 3000);
        
    } catch (error) {
        console.error('❌ Erreur:', error);
        showLoading(false);
        alert('Erreur chargement: ' + error.message);
    }
}

// ==================== PRÉDICTION ONNX ====================
async function predictFromImage(imageElement) {
    if (!session) return null;
    
    // Redimensionner
    const canvas = document.createElement('canvas');
    canvas.width = IMG_SIZE;
    canvas.height = IMG_SIZE;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(imageElement, 0, 0, IMG_SIZE, IMG_SIZE);
    
    // Récupérer pixels
    const imageData = ctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE);
    const pixels = imageData.data;
    
    // Format ONNX: [1, 3, 224, 224]
    const input = new Float32Array(1 * 3 * IMG_SIZE * IMG_SIZE);
    
    for (let y = 0; y < IMG_SIZE; y++) {
        for (let x = 0; x < IMG_SIZE; x++) {
            const idx = (y * IMG_SIZE + x) * 4;
            const r = (pixels[idx] / 255 - 0.485) / 0.229;
            const g = (pixels[idx + 1] / 255 - 0.456) / 0.224;
            const b = (pixels[idx + 2] / 255 - 0.406) / 0.225;
            
            input[0 * IMG_SIZE * IMG_SIZE + y * IMG_SIZE + x] = r;
            input[1 * IMG_SIZE * IMG_SIZE + y * IMG_SIZE + x] = g;
            input[2 * IMG_SIZE * IMG_SIZE + y * IMG_SIZE + x] = b;
        }
    }
    
    // Inférence
    const tensor = new ort.Tensor('float32', input, [1, 3, IMG_SIZE, IMG_SIZE]);
    const feeds = { input: tensor };
    const results = await session.run(feeds);
    const outputData = results.output.data;
    
    // Softmax
    const probs = softmax(Array.from(outputData));
    const bestIdx = probs.indexOf(Math.max(...probs));
    const sku = mapping[bestIdx] || 'unknown';
    const confidence = probs[bestIdx];
    const product = skuMap[sku];
    
    return { sku, confidence, product };
}

function softmax(arr) {
    const max = Math.max(...arr);
    const exps = arr.map(v => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(v => v / sum);
}

// ==================== AFFICHAGE ====================
function showResult(sku, confidence, product) {
    const resultCard = document.getElementById('result-card');
    const resultTitle = document.getElementById('result-title');
    const resultDetails = document.getElementById('result-details');
    
    resultTitle.textContent = `${sku} (${(confidence * 100).toFixed(2)}%)`;
    
    if (product) {
        resultDetails.innerHTML = `
            🛒 ${product.product_name}<br>
            🏷️ Marque: ${product.brand}<br>
            📦 Catégorie: ${product.category}<br>
            📏 Capacité: ${product.capacity}<br>
            📦 Emballage: ${product.emballage}<br>
            🍓 Saveur: ${product.saveur}
        `;
    } else {
        resultDetails.textContent = '❌ Produit non trouvé';
    }
    
    resultCard.style.display = 'block';
    setTimeout(() => resultCard.scrollIntoView({ behavior: 'smooth' }), 100);
}

function hideResult() {
    document.getElementById('result-card').style.display = 'none';
}

// ==================== MODE PHOTO ====================
async function takePhoto() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        const video = document.createElement('video');
        video.srcObject = stream;
        video.play();
        
        setTimeout(async () => {
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);
            
            const img = new Image();
            img.src = canvas.toDataURL();
            img.onload = async () => {
                const prediction = await predictFromImage(img);
                if (prediction) {
                    showResult(prediction.sku, prediction.confidence, prediction.product);
                    const preview = document.getElementById('photo-preview');
                    preview.innerHTML = '';
                    preview.appendChild(img);
                }
                stream.getTracks().forEach(t => t.stop());
            };
        }, 500);
    } catch (error) {
        alert('Erreur caméra: ' + error.message);
    }
}

// ==================== MODE GALERIE ====================
document.getElementById('choose-image-btn').onclick = () => {
    document.getElementById('gallery-input').click();
};

document.getElementById('gallery-input').onchange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    const img = new Image();
    img.src = URL.createObjectURL(file);
    img.onload = async () => {
        const prediction = await predictFromImage(img);
        if (prediction) {
            showResult(prediction.sku, prediction.confidence, prediction.product);
            const preview = document.getElementById('gallery-preview');
            preview.innerHTML = '';
            preview.appendChild(img);
        }
    };
};

// ==================== MODE STREAM ====================
async function startStream() {
    if (streamActive) return;
    
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        streamVideo = document.getElementById('stream-video');
        streamVideo.srcObject = stream;
        streamTrack = stream.getTracks()[0];
        
        streamActive = true;
        lastStableSku = null;
        
        document.getElementById('start-stream-btn').style.display = 'none';
        document.getElementById('stop-stream-btn').style.display = 'block';
        document.getElementById('stream-status').textContent = '🔍 Stabilisation...';
        document.getElementById('stream-status').className = 'status-badge orange';
        
        streamTimer = setInterval(async () => {
            if (!streamActive) return;
            
            const canvas = document.createElement('canvas');
            canvas.width = streamVideo.videoWidth;
            canvas.height = streamVideo.videoHeight;
            canvas.getContext('2d').drawImage(streamVideo, 0, 0);
            
            const img = new Image();
            img.src = canvas.toDataURL();
            img.onload = async () => {
                const prediction = await predictFromImage(img);
                if (prediction) {
                    const predCard = document.getElementById('stream-prediction');
                    predCard.style.display = 'block';
                    predCard.innerHTML = `
                        <div class="prediction-sku">${prediction.sku}</div>
                        <div class="prediction-name">${prediction.product?.product_name || 'Inconnu'}</div>
                        <div class="prediction-confidence">Confiance: ${(prediction.confidence * 100).toFixed(1)}%</div>
                    `;
                    
                    if (lastStableSku === prediction.sku) {
                        if (streamStabilizationTimer) clearTimeout(streamStabilizationTimer);
                        streamStabilizationTimer = setTimeout(() => {
                            if (streamActive) {
                                showResult(prediction.sku, prediction.confidence, prediction.product);
                                stopStream();
                            }
                        }, STABILIZATION_SECONDS * 1000);
                    } else {
                        lastStableSku = prediction.sku;
                        if (streamStabilizationTimer) clearTimeout(streamStabilizationTimer);
                        streamStabilizationTimer = setTimeout(() => {
                            if (streamActive) {
                                showResult(prediction.sku, prediction.confidence, prediction.product);
                                stopStream();
                            }
                        }, STABILIZATION_SECONDS * 1000);
                    }
                }
            };
        }, STREAM_INTERVAL_MS);
        
    } catch (error) {
        alert('Erreur caméra: ' + error.message);
    }
}

function stopStream() {
    streamActive = false;
    if (streamTimer) clearInterval(streamTimer);
    if (streamStabilizationTimer) clearTimeout(streamStabilizationTimer);
    if (streamTrack) streamTrack.stop();
    
    document.getElementById('start-stream-btn').style.display = 'block';
    document.getElementById('stop-stream-btn').style.display = 'none';
    document.getElementById('stream-status').textContent = 'Arrêté';
    document.getElementById('stream-status').className = 'status-badge';
    document.getElementById('stream-prediction').style.display = 'none';
}

// ==================== MODE INVENTAIRE ====================
async function startInventory() {
    if (inventoryActive) return;
    
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        inventoryVideo = document.getElementById('inventory-video');
        inventoryVideo.srcObject = stream;
        inventoryTrack = stream.getTracks()[0];
        
        inventoryActive = true;
        inventoryLastStableSku = null;
        currentInventoryPrediction = null;
        
        document.getElementById('start-inventory-btn').style.display = 'none';
        document.getElementById('stop-inventory-btn').style.display = 'block';
        document.getElementById('inventory-status').textContent = '🔍 Stabilisation...';
        document.getElementById('inventory-status').className = 'status-badge orange';
        
        inventoryTimer = setInterval(async () => {
            if (!inventoryActive) return;
            
            const canvas = document.createElement('canvas');
            canvas.width = inventoryVideo.videoWidth;
            canvas.height = inventoryVideo.videoHeight;
            canvas.getContext('2d').drawImage(inventoryVideo, 0, 0);
            
            const img = new Image();
            img.src = canvas.toDataURL();
            img.onload = async () => {
                const prediction = await predictFromImage(img);
                if (prediction) {
                    currentInventoryPrediction = prediction;
                    
                    const predCard = document.getElementById('inventory-prediction');
                    predCard.style.display = 'block';
                    predCard.innerHTML = `
                        <div class="prediction-sku">${prediction.sku}</div>
                        <div class="prediction-name">${prediction.product?.product_name || 'Inconnu'}</div>
                        <div class="prediction-confidence">Confiance: ${(prediction.confidence * 100).toFixed(1)}%</div>
                        <button id="add-inventory-btn" class="btn-primary" style="margin-top: 12px;">➕ Ajouter</button>
                    `;
                    
                    document.getElementById('add-inventory-btn').onclick = () => addToInventory();
                    
                    if (inventoryLastStableSku === prediction.sku) {
                        if (inventoryStabilizationTimer) clearTimeout(inventoryStabilizationTimer);
                        inventoryStabilizationTimer = setTimeout(() => {
                            if (inventoryActive) {
                                document.getElementById('inventory-status').textContent = '✅ Produit détecté! Cliquez sur Ajouter';
                                document.getElementById('inventory-status').className = 'status-badge green';
                            }
                        }, STABILIZATION_SECONDS * 1000);
                    } else {
                        inventoryLastStableSku = prediction.sku;
                        if (inventoryStabilizationTimer) clearTimeout(inventoryStabilizationTimer);
                        inventoryStabilizationTimer = setTimeout(() => {
                            if (inventoryActive) {
                                document.getElementById('inventory-status').textContent = '✅ Produit détecté! Cliquez sur Ajouter';
                                document.getElementById('inventory-status').className = 'status-badge green';
                            }
                        }, STABILIZATION_SECONDS * 1000);
                    }
                }
            };
        }, STREAM_INTERVAL_MS);
        
    } catch (error) {
        alert('Erreur caméra: ' + error.message);
    }
}

function addToInventory() {
    if (!currentInventoryPrediction) return;
    
    const sku = currentInventoryPrediction.sku;
    inventoryCount[sku] = (inventoryCount[sku] || 0) + 1;
    inventoryList.push({ ...currentInventoryPrediction });
    
    updateInventoryDisplay();
    
    currentInventoryPrediction = null;
    inventoryLastStableSku = null;
    document.getElementById('inventory-prediction').style.display = 'none';
    document.getElementById('inventory-status').textContent = '🔍 Scannez le prochain produit...';
    document.getElementById('inventory-status').className = 'status-badge orange';
}

function stopInventory() {
    inventoryActive = false;
    if (inventoryTimer) clearInterval(inventoryTimer);
    if (inventoryStabilizationTimer) clearTimeout(inventoryStabilizationTimer);
    if (inventoryTrack) inventoryTrack.stop();
    
    document.getElementById('start-inventory-btn').style.display = 'block';
    document.getElementById('stop-inventory-btn').style.display = 'none';
    document.getElementById('inventory-status').textContent = 'Arrêté';
    document.getElementById('inventory-status').className = 'status-badge';
    document.getElementById('inventory-prediction').style.display = 'none';
}

function updateInventoryDisplay() {
    const container = document.getElementById('inventory-list');
    const totalItems = Object.values(inventoryCount).reduce((a, b) => a + b, 0);
    
    if (inventoryList.length === 0) {
        container.style.display = 'none';
        return;
    }
    
    container.style.display = 'block';
    
    let html = `
        <div class="inventory-header">
            <h4>📦 Inventaire (${totalItems} articles)</h4>
            <span class="inventory-count">${inventoryList.length} produits</span>
        </div>
    `;
    
    const uniqueItems = {};
    inventoryList.forEach(item => {
        if (!uniqueItems[item.sku]) uniqueItems[item.sku] = item;
    });
    
    Object.values(uniqueItems).forEach(item => {
        const count = inventoryCount[item.sku];
        const product = item.product;
        html += `
            <div class="inventory-item">
                <div class="inventory-item-count">${count}</div>
                <div class="inventory-item-info">
                    <div class="inventory-item-name">${product?.product_name || item.sku}</div>
                    <div class="inventory-item-detail">${product?.brand || ''} | ${product?.capacity || ''}</div>
                </div>
                <div class="inventory-item-qty">${count}x</div>
            </div>
        `;
    });
    
    container.innerHTML = html;
}

// ==================== MODE SWITCH ====================
document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.onclick = () => {
        document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        document.querySelectorAll('.mode-panel').forEach(panel => panel.classList.remove('active'));
        document.getElementById(`${btn.dataset.mode}-mode`).classList.add('active');
        
        hideResult();
        
        if (streamActive) stopStream();
        if (inventoryActive) stopInventory();
    };
});

// ==================== INIT ====================
document.getElementById('take-photo-btn').onclick = takePhoto;
document.getElementById('start-stream-btn').onclick = startStream;
document.getElementById('stop-stream-btn').onclick = stopStream;
document.getElementById('start-inventory-btn').onclick = startInventory;
document.getElementById('stop-inventory-btn').onclick = stopInventory;

function showLoading(show) {
    document.getElementById('loading-overlay').style.display = show ? 'flex' : 'none';
}

// Démarrer
loadModel();