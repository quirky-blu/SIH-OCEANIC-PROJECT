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
import re

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

# GitHub Models configuration
AZURE_ENDPOINT = "https://models.github.ai/inference"
GPT_MODEL = "gpt-4o"  # Updated to a more reliable model

# Load environment variables
load_dotenv()
current_dir = Path(__file__).parent
env_file = current_dir / '.env'
load_dotenv(env_file)

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN_OLD")

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

def smart_file_matcher(prompt: str, available_files: List[str]) -> List[str]:
    """
    Fallback function to match files based on keywords if AI fails
    """
    prompt_lower = prompt.lower()
    matched_files = []
    
    # Define keyword mappings
    keyword_mappings = {
        # 'pollution': ['high_pollution_areas','extremely_high_pollution_areas']
        # 'shipwrecks' : ['shipwrecks']
        # 'resources' : ['petroleum','magnesium'] # 3 aur add kar lena
        # 'oil' : ['petroleum']
        # 'ores' : ['magnesium']

        'fish': ['Thunnus_albacares', 'Sardinella_longiceps', 'Stolephorus_indicus', 'Rastrelliger_kanagurta'],
        'tuna': ['Thunnus_albacares'],
        'sardine': ['Sardinella_longiceps'],
        'anchovy': ['Stolephorus_indicus'],
        'mackerel': ['Rastrelliger_kanagurta'],
        'algae': ['Hypnea_musciformis', 'Ulva_lactuca', 'Noctiluca_scintillans'],
        'plankton': ['Noctiluca_scintillans', 'Proboscia_alata', 'Rhizosolenia_hebetata', 'Thalassionema_nitzschioides', 'Tripos_furca'],
        'copepod': ['Conchoecetta_giesbrechti', 'Euconchoecia_aculeata', 'Metaconchoecia_rotundata', 'Orthoconchoecia_atlantica', 'Proceroecia_procera'],
        'seaweed': ['Hypnea_musciformis', 'Ulva_lactuca'],
        'clownfish': ['Amphiprion_clarkii']
    }
    
    # Check for specific keywords
    for keyword, files in keyword_mappings.items():
        if keyword in prompt_lower:
            for file in files:
                if file in available_files:
                    matched_files.append(file)
    
    # If asking for "one type" or "any one", return just one file
    if any(phrase in prompt_lower for phrase in ['one type', 'any one', 'single', 'just one']):
        if 'fish' in prompt_lower:
            return ['Sardinella_longiceps'] if 'Sardinella_longiceps' in available_files else []
        elif matched_files:
            return [matched_files[0]]
    
    return matched_files

async def query_gpt4(prompt: str, available_files: List[str]) -> Dict[str, Any]:
    """Query GPT-4 via GitHub Models to determine which files to use and how to process them"""

    system_prompt = f"""
You are an expert at analyzing natural language queries about marine geospatial data and determining which GeoJSON files to query.

Available GeoJSON files: {', '.join(available_files)}

File descriptions:
- high_pollution_areas.geojson: Areas with high pollution/harmful algal blooms
- Thunnus_albacares.geojson: Yellowfin tuna distribution
- Sardinella_longiceps.geojson: Indian oil sardine distribution  
- Stolephorus_indicus.geojson: Indian anchovy distribution
- Rastrelliger_kanagurta.geojson: Indian mackerel distribution
- Amphiprion_clarkii.geojson: Clark's anemonefish/clownfish distribution
- Hypnea_musciformis.geojson: Red seaweed species
- Ulva_lactuca.geojson: Green seaweed species
- Noctiluca_scintillans.geojson: Bioluminescent dinoflagellate plankton
- Other files: Various marine species and plankton

CRITICAL RULES:
1. Be VERY SELECTIVE - only choose files that directly match the query
2. If user asks for "one type" or "any one", return exactly ONE file
3. If asking for pollution, ONLY return high_pollution_areas.geojson
4. If asking for fish, only return fish species files (Thunnus, Sardinella, etc.)
5. NEVER return all files unless explicitly asked for "all data" or "everything"
6. Default to returning 1-3 files maximum

Your task:
1. Classify query as LOCATION-BASED (coordinates mentioned) or GLOBAL (general search)
2. Select ONLY the most relevant files - be strict!
3. Provide short description
4. Extract keywords

Respond ONLY in valid JSON format:
{{
    "query_type": "location_based" or "global",
    "files_to_query": ["exact_filename_from_list"],
    "response_description": "Brief description",
    "search_terms": ["keyword1", "keyword2"]
}}

Examples:
Query: "Show high pollution areas"
{{
    "query_type": "global",
    "files_to_query": ["high_pollution_areas"],
    "response_description": "Displaying high pollution areas",
    "search_terms": ["pollution", "algal blooms"]
}}

Query: "Show any one type of fish"
{{
    "query_type": "global", 
    "files_to_query": ["Sardinella_longiceps"],
    "response_description": "Showing Indian oil sardine distribution",
    "search_terms": ["fish", "sardine"]
}}
"""

    try:
        client = ChatCompletionsClient(
            endpoint=AZURE_ENDPOINT,
            credential=AzureKeyCredential(GITHUB_TOKEN),
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query: {prompt}"}
        ]

        response = client.complete(
            messages=messages,
            temperature=0.1,
            model=GPT_MODEL,
            max_tokens=500
        )

        ai_response = response.choices[0].message.content
        print(f"AI Response: {ai_response}")

        # Clean the response - remove any markdown formatting
        cleaned_response = ai_response.strip()
        if cleaned_response.startswith('```json'):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.endswith('```'):
            cleaned_response = cleaned_response[:-3]
        cleaned_response = cleaned_response.strip()

        try:
            parsed_response = json.loads(cleaned_response)
            
            # Validate the response
            if not isinstance(parsed_response.get('files_to_query'), list):
                raise ValueError("files_to_query must be a list")
            
            # Ensure filenames don't have .geojson extension
            files_to_query = []
            for file in parsed_response.get('files_to_query', []):
                clean_filename = file.replace('.geojson', '')
                if clean_filename in available_files:
                    files_to_query.append(clean_filename)
            
            parsed_response['files_to_query'] = files_to_query
            
            # If AI didn't find good matches, use fallback
            if not files_to_query:
                fallback_files = smart_file_matcher(prompt, available_files)
                if fallback_files:
                    parsed_response['files_to_query'] = fallback_files
                    parsed_response['response_description'] = f"Using fallback matching for: {prompt}"
            
            return parsed_response
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"JSON parsing error: {e}, AI response: {ai_response}")
            # Use fallback matching
            fallback_files = smart_file_matcher(prompt, available_files)
            return {
                "query_type": "global",
                "files_to_query": fallback_files or (available_files[:1] if available_files else []),
                "response_description": f"Fallback matching for: {prompt}",
                "search_terms": prompt.lower().split()[:3]
            }

    except Exception as e:
        print(f"Error querying AI: {e}")
        # Use fallback matching
        fallback_files = smart_file_matcher(prompt, available_files)
        return {
            "query_type": "global",
            "files_to_query": fallback_files or (available_files[:1] if available_files else []),
            "response_description": f"Error occurred, using fallback for: {prompt}",
            "search_terms": prompt.lower().split()[:3]
        }

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
    """Test endpoint with fallback logic"""
    try:
        print(f"Test query received: {request.prompt}")

        # Use smart file matcher for testing
        matched_files = smart_file_matcher(request.prompt, available_files)
        
        # If no matches, take first file as fallback
        if not matched_files:
            matched_files = available_files[:1]

        result_data = {}

        for filename in matched_files:
            if filename in geojson_data:
                file_data = geojson_data[filename]
                if request.coordinates:
                    filtered_data = filter_geojson_by_bounds(file_data, request.coordinates)
                    if filtered_data['features']:
                        result_data[filename] = filtered_data
                else:
                    result_data[filename] = file_data

        return {
            "status": "success",
            "query_analysis": {
                "query_type": "location_based" if request.coordinates else "global",
                "files_to_query": matched_files,
                "response_description": f"Test query for: {request.prompt}",
                "search_terms": request.prompt.lower().split()[:3]
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
        raise HTTPException(status_code=500, detail=f"Test query error: {str(e)}")

@app.post("/query")
async def query_geojson(request: QueryRequest):
    """Main endpoint to query GeoJSON data based on natural language prompt"""

    if not geojson_data:
        raise HTTPException(status_code=500, detail="No GeoJSON data loaded")

    try:
        print(f"Processing query: {request.prompt}")
        
        # Query AI to understand the request
        ai_analysis = await query_gpt4(request.prompt, available_files)
        print(f"AI Analysis: {ai_analysis}")

        query_type = ai_analysis.get('query_type', 'global')
        files_to_query = ai_analysis.get('files_to_query', [])

        # Ensure we have valid files
        valid_files = [f for f in files_to_query if f in geojson_data]

        if not valid_files:
            print("No valid files found, using fallback")
            fallback_files = smart_file_matcher(request.prompt, available_files)
            valid_files = fallback_files or available_files[:1]

        print(f"Valid files to query: {valid_files}")

        result_data = {}

        for filename in valid_files:
            if filename in geojson_data:
                file_data = geojson_data[filename]

                if query_type == "location_based" and request.coordinates:
                    filtered_data = filter_geojson_by_bounds(file_data, request.coordinates)
                    if filtered_data['features']:
                        result_data[filename] = filtered_data
                else:
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
        print(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/query-stream")
async def query_geojson_stream(request: QueryRequest):
    """Stream GeoJSON data - placeholder for streaming functionality"""
    return await query_geojson(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)