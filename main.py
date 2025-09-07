from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import json
import os
import glob
from pathlib import Path
import httpx
import asyncio
from dotenv import load_dotenv

from openai import OpenAI
from dotenv import load_dotenv
# Make sure your new GitHub PAT is set as an env variable:
# export GITHUB_TOKEN=ghp_xxx...
# or put it in a .env and load before running.
load_dotenv()

client = OpenAI(
    base_url="https://models.github.ai/inference",
    api_key=os.getenv("GITHUB_TOKEN")
)

current_dir = Path(__file__).parent
env_file = current_dir / '.env'
load_dotenv(env_file)

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
# Load environment variables


app = FastAPI(title="GeoJSON Query API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class QueryRequest(BaseModel):
    prompt: str
    coordinates: Optional[Dict[str, float]] = None  # {"north": float, "south": float, "east": float, "west": float}

# Global variables to store loaded GeoJSON data
geojson_data = {}
available_files = []


def load_geojson_files():
    """Load all GeoJSON files from the resources folder"""
    global geojson_data, available_files
    
    resources_path = Path("resources")
    if not resources_path.exists():
        print(f"Warning: Resources folder not found at {resources_path}")
        return
    
    geojson_files = glob.glob(str(resources_path / "*.geojson"))
    available_files = [os.path.basename(f).replace('.geojson', '') for f in geojson_files]
    
    for file_path in geojson_files:
        filename = os.path.basename(file_path).replace('.geojson', '')
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                geojson_data[filename] = json.load(f)
            print(f"Loaded: {filename}.geojson")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    print(f"Total files loaded: {len(geojson_data)}")

def filter_geojson_by_bounds(geojson, bounds):
    """Filter GeoJSON features by coordinate bounds"""
    filtered_features = []
    
    for feature in geojson.get('features', []):
        geometry = feature.get('geometry', {})
        if geometry.get('type') == 'Point':
            coords = geometry.get('coordinates', [])
            if len(coords) >= 2:
                lon, lat = coords[0], coords[1]
                if (bounds['west'] <= lon <= bounds['east'] and 
                    bounds['south'] <= lat <= bounds['north']):
                    filtered_features.append(feature)
        elif geometry.get('type') in ['LineString', 'MultiLineString', 'Polygon', 'MultiPolygon']:
            # For complex geometries, check if any coordinate falls within bounds
            # This is a simplified approach - you might want more sophisticated geometry intersection
            coords = extract_all_coordinates(geometry)
            if any(bounds['west'] <= coord[0] <= bounds['east'] and 
                   bounds['south'] <= coord[1] <= bounds['north'] for coord in coords):
                filtered_features.append(feature)
    
    return {
        "type": "FeatureCollection",
        "features": filtered_features
    }

def extract_all_coordinates(geometry):
    """Extract all coordinates from a geometry recursively"""
    coords = []
    geometry_type = geometry.get('type')
    coordinates = geometry.get('coordinates', [])
    
    if geometry_type == 'Point':
        coords.append(coordinates)
    elif geometry_type in ['LineString', 'MultiPoint']:
        coords.extend(coordinates)
    elif geometry_type in ['Polygon', 'MultiLineString']:
        for ring in coordinates:
            coords.extend(ring)
    elif geometry_type == 'MultiPolygon':
        for polygon in coordinates:
            for ring in polygon:
                coords.extend(ring)
    
    return coords

GPT_MODEL = "openai/gpt-4o"


async def query_gpt4(prompt: str, available_files: List[str]) -> Dict[str, Any]:
    """Query GPT-4 via GitHub Models to determine which files to use and how to process them"""
    
    folder_list_placeholder = [
        "Thunnus_albacares.geojson","Aidanosagitta_regularis.geojson","Charybdis__Archias__smithii.geojson",
        "Conchoecetta_giesbrechti.geojson","Cossura_coasta.geojson","Cypridina_dentata.geojson",
        "Euconchoecia_aculeata.geojson","Flaccisagitta_enflata.geojson","Hypnea_musciformis.geojson",
        "Amphiprion_clarkii.geojson","Sardinella_longiceps.geojson","Metaconchoecia_rotundata.geojson",
        "Noctiluca_scintillans.geojson","Orthoconchoecia_atlantica.geojson","Stolephorus_indicus.geojson",
        "Proboscia_alata.geojson","Proceroecia_procera.geojson","Pseudanchialina_pusilla.geojson",
        "Pterosagitta_draco.geojson","Rhizosolenia_hebetata.geojson","Serratosagitta_pacifica.geojson",
        "Rastrelliger_kanagurta.geojson","Siriella_gracilis.geojson","Thalassionema_nitzschioides.geojson",
        "Tripos_furca.geojson","Ulva_lactuca.geojson"
    ]
    
    system_prompt = f"""You are an expert at analyzing natural language queries about geospatial data and determining which GeoJSON files to query.
{folder_list_placeholder}
Available files: {', '.join(available_files)}
Based on the user's natural language prompt, you need to:
1. Determine if this is a LOCATION-BASED query (show data within specific coordinates) or a GLOBAL query (find locations across entire dataset)
2. Identify which GeoJSON file(s) should be queried
3. Provide a clear response about what data to return
Respond in JSON format with:
{{
    "query_type": "location_based" or "global",
    "files_to_query": ["filename1", "filename2"],
    "response_description": "Description of what data should be returned",
    "search_terms": ["relevant", "keywords"]
}}"""
    
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            model=GPT_MODEL,
            temperature=0.1,
            max_tokens=500
        )

        ai_response = response.choices[0].message.content

        try:
            return json.loads(ai_response)
        except json.JSONDecodeError:
            return {
                "query_type": "global",
                "files_to_query": available_files[:3] if available_files else [],
                "response_description": f"Could not parse AI response: {ai_response}",
                "search_terms": []
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying GPT-4 API: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load GeoJSON files on startup"""
    load_geojson_files()
    if not GITHUB_TOKEN:
        print("Warning: GITHUB_TOKEN not found in environment variables")

@app.get("/")
async def root():
    return {
        "message": "GeoJSON Query API is running",
        "available_files": available_files,
        "total_files": len(geojson_data),
        "github_configured": bool(GITHUB_TOKEN)
    }


@app.get("/files")
async def list_files():
    """Get list of available GeoJSON files"""
    return {
        "available_files": available_files,
        "file_details": {name: len(data.get('features', [])) for name, data in geojson_data.items()}
    }

@app.post("/test-query")
async def test_query(request: QueryRequest):
    """Test endpoint with simpler logic"""
    try:
        print(f"Test query received: {request.prompt}")
        
        # Simple test response
        result_data = {}
        
        # Take first 3 files for testing
        test_files = list(geojson_data.keys())[:3]
        
        for filename in test_files:
            file_data = geojson_data[filename]
            if request.coordinates:
                # Filter by coordinates
                filtered_data = filter_geojson_by_bounds(file_data, request.coordinates)
                if filtered_data['features']:
                    result_data[filename] = filtered_data
            else:
                result_data[filename] = file_data
        
        return {
            "status": "success",
            "query_analysis": {
                "query_type": "location_based" if request.coordinates else "global",
                "files_to_query": test_files,
                "response_description": "Test query response",
                "search_terms": []
            },
            "query_type": "location_based" if request.coordinates else "global",
            "coordinates_used": request.coordinates,
            "files_queried": list(result_data.keys()),
            "data": result_data,
            "summary": {
                "total_files": len(result_data),
                "total_features": sum(len(data.get('features', [])) for data in result_data.values())
            }
        }
        
    except Exception as e:
        print(f"Test query error: {e}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Test query error: {str(e)}")

@app.post("/query")
async def query_geojson(request: QueryRequest):
    """Main endpoint to query GeoJSON data based on natural language prompt"""
    
    if not geojson_data:
        raise HTTPException(status_code=500, detail="No GeoJSON data loaded")
    
    if not GITHUB_TOKEN:
        raise HTTPException(status_code=500, detail="GitHub token not configured")
    
    try:
        # Query GPT-4 to understand the request
        ai_analysis = await query_gpt4(request.prompt, available_files)
        
        # Rest of the function remains the same...
        query_type = ai_analysis.get('query_type', 'global')
        files_to_query = ai_analysis.get('files_to_query', [])
        
        # Filter files that actually exist
        valid_files = [f for f in files_to_query if f in geojson_data]
        
        if not valid_files:
            # If no specific files identified, use all available files
            valid_files = list(geojson_data.keys())
        
        result_data = {}
        
        for filename in valid_files:
            file_data = geojson_data[filename]
            
            if query_type == "location_based" and request.coordinates:
                # Filter by coordinates for location-based queries
                filtered_data = filter_geojson_by_bounds(file_data, request.coordinates)
                if filtered_data['features']:  # Only include if has features
                    result_data[filename] = filtered_data
            else:
                # For global queries, return full dataset
                result_data[filename] = file_data
        
        return {
            "status": "success",
            "query_analysis": ai_analysis,
            "query_type": query_type,
            "coordinates_used": request.coordinates if query_type == "location_based" else None,
            "files_queried": list(result_data.keys()),
            "data": result_data,
            "summary": {
                "total_files": len(result_data),
                "total_features": sum(len(data.get('features', [])) for data in result_data.values())
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/query-stream")
async def query_geojson_stream(request: QueryRequest):
    """Stream GeoJSON data - this endpoint returns the same data but could be modified for actual streaming"""
    # For true streaming, you would use FastAPI's StreamingResponse
    # This is a placeholder that returns the same data as the regular query endpoint
    return await query_geojson(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)