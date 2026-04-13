# Diffusion Tutorial Blog

## Running locally

Install `http-server` if you don't have it:

```bash
npm install -g http-server
```

Then from this directory:

```bash
http-server
```

Open [http://localhost:8080](http://localhost:8080) in your browser.

## Generating media

Install dependencies:

```bash
pip install -r requirements.txt
```

Run all scripts to regenerate the images and animations:

These are the scripts that their files are ready to use:

```bash
python "distribution transport.py"
python score-matching/langevine_dynamics.py
```
