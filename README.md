# RealSense D455 — Captura RGB-D

Pipeline de captura de pares RGB + depth bruto para a Intel RealSense D455, compatível com HorizonGS.

---

## Pré-requisitos no host

O módulo de kernel do RealSense **precisa estar instalado na máquina host** — o container Docker não consegue fazer isso por si só.

```bash
# Adiciona o repositório Intel
sudo mkdir -p /etc/apt/keyrings
curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp \
    | sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null

echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] \
https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" \
    | sudo tee /etc/apt/sources.list.d/librealsense.list

sudo apt-get update

# Instala driver de kernel + utilitários
sudo apt-get install librealsense2-dkms librealsense2-utils
```

Verifique que a câmera está sendo reconhecida antes de continuar:

```bash
rs-enumerate-devices
```

---

## Pipeline completo

### 1. Construir a imagem Docker

Feito uma única vez (ou quando o código mudar):

```bash
./docker-run.sh build
```

### 2. (Opcional) Diagnosticar a câmera

Testa quais combinações de resolução/FPS a D455 aceita na porta USB atual:

```bash
./docker-run.sh probe
```

### 3. Capturar os dados

Escolha **um** dos dois modos abaixo.

#### Modo A — Captura direta

Abre o preview ao vivo. Pressione `ESPAÇO` para salvar cada par RGB+depth e `Q` para encerrar.

```bash
./docker-run.sh capture
```

Os frames são salvos em `output/` imediatamente.

#### Modo B — Gravar .bag e extrair depois (recomendado)

Grava a sessão inteira num arquivo `.bag` com os streams brutos (RGB + depth z16 sem colorização). Depois, extrai todos os frames de uma vez.

**Etapa 1 — gravar:**
```bash
# Nome automático (sessao_YYYYMMDD_HHMMSS.bag)
./docker-run.sh bag

# Ou com nome específico
./docker-run.sh bag captura01.bag
```

Pressione `Q` ou `Ctrl+C` para parar a gravação.

**Etapa 2 — extrair frames do .bag:**
```bash
./docker-run.sh extract sessao_20260430_143000.bag
```

### 4. Resultado

Após captura ou extração, a pasta `output/` terá:

```
output/
├── rgb/
│   ├── frame_00001.jpg
│   ├── frame_00002.jpg
│   └── ...
├── depth/
│   ├── frame_00001.npy   ← uint16 bruto (valores z16 do sensor)
│   ├── frame_00002.npy
│   └── ...
└── meta.json             ← depth_scale e intrínsecos da câmera
```

Para converter depth para metros em Python:

```python
import numpy as np, json

meta  = json.load(open("output/meta.json"))
scale = meta["depth_scale_m_per_unit"]

depth_u16 = np.load("output/depth/frame_00001.npy")   # uint16
depth_m   = depth_u16.astype(float) * scale            # float64, metros
```

---

## Sem Docker (ambiente local)

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt

python realsense_capture.py             # captura direta
python realsense_capture.py --bag       # grava .bag
python realsense_capture.py --extract sessao.bag
```

---

## Referência rápida

| Comando | Descrição |
|---|---|
| `./docker-run.sh build` | Constrói a imagem Docker |
| `./docker-run.sh probe` | Diagnóstico da câmera |
| `./docker-run.sh capture` | Captura direta (ESPAÇO=salvar) |
| `./docker-run.sh bag [arquivo]` | Grava sessão em .bag |
| `./docker-run.sh extract arquivo.bag` | Extrai frames de um .bag |
