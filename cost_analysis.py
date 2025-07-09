import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple
import math

# API Configuration
GEMINI_API_KEY = "AIzaSyB46mW-7p4MIrKSe-oudQLpjxWli6XjVpE"
GEMINI_MODEL = "gemini-2.0-flash-001"

WORLDREF_HUBS = {
    "Mumbai": {"lat": 19.0760, "lng": 72.8777, "port": "JNPT", "specialization": "Middle East, Europe"},
    "Chennai": {"lat": 13.0827, "lng": 80.2707, "port": "Chennai Port", "specialization": "Southeast Asia"},
    "Kolkata": {"lat": 22.5726, "lng": 88.3639, "port": "Kolkata Port", "specialization": "Bangladesh, Myanmar"},
    "Delhi": {"lat": 28.6139, "lng": 77.2090, "port": "Main Hub", "specialization": "Inland Hub"}
}

PRICING_CONFIG = {
    "base_rate_per_km_per_ton": 15,
    "weight_factors": {
        "< 1 ton": 1.2,
        "1-5 tons": 1.0,
        "5-10 tons": 0.9,
        "> 10 tons": 0.8
    },
    "commodity_factors": {
        "General Goods": 1.0,
        "Hazardous Materials": 1.5,
        "Fragile Items": 1.3,
        "Perishable Goods": 1.4,
        "Electronics": 1.2
    },
    "port_charges": {
        "Mumbai": 10000,
        "Chennai": 9000,
        "Kolkata": 8500
    },
    "partner_rates": {
        "Express": 220,
        "Standard": 150,
        "Economy": 100
    }
}

INDIAN_CITIES = [
    "Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata", "Hyderabad",
    "Pune", "Ahmedabad", "Surat", "Jaipur", "Lucknow", "Kanpur",
    "Nagpur", "Indore", "Bhopal", "Visakhapatnam", "Kochi", "Coimbatore",
    "Gurgaon", "Noida", "Faridabad", "Ghaziabad", "Agra", "Meerut",
    "Varanasi", "Allahabad", "Patna", "Ranchi", "Bhubaneswar", "Cuttack"
]

DESTINATION_COUNTRIES = [
    "United States", "United Kingdom", "Germany", "France", "Japan",
    "China", "Singapore", "UAE", "Saudi Arabia", "Australia",
    "Canada", "Netherlands", "Italy", "Spain", "South Korea",
    "Bangladesh", "Myanmar", "Sri Lanka", "Nepal", "Maldives"
]

# Major cities by country for destination
DESTINATION_CITIES = {
    "United States": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose"],
    "United Kingdom": ["London", "Birmingham", "Manchester", "Leeds", "Glasgow", "Liverpool", "Newcastle", "Sheffield", "Bristol", "Edinburgh"],
    "Germany": ["Berlin", "Hamburg", "Munich", "Cologne", "Frankfurt", "Stuttgart", "Düsseldorf", "Dortmund", "Essen", "Bremen"],
    "France": ["Paris", "Marseille", "Lyon", "Toulouse", "Nice", "Nantes", "Strasbourg", "Montpellier", "Bordeaux", "Lille"],
    "Japan": ["Tokyo", "Osaka", "Yokohama", "Nagoya", "Sapporo", "Fukuoka", "Kobe", "Kyoto", "Kawasaki", "Saitama"],
    "China": ["Shanghai", "Beijing", "Shenzhen", "Guangzhou", "Chengdu", "Tianjin", "Nanjing", "Wuhan", "Xi'an", "Hangzhou"],
    "Singapore": ["Singapore"],
    "UAE": ["Dubai", "Abu Dhabi", "Sharjah", "Ajman", "Fujairah", "Ras Al Khaimah", "Umm Al Quwain"],
    "Saudi Arabia": ["Riyadh", "Jeddah", "Mecca", "Medina", "Dammam", "Khobar", "Tabuk", "Buraidah", "Khamis Mushait", "Hail"],
    "Australia": ["Sydney", "Melbourne", "Brisbane", "Perth", "Adelaide", "Gold Coast", "Newcastle", "Canberra", "Sunshine Coast", "Wollongong"],
    "Canada": ["Toronto", "Montreal", "Vancouver", "Calgary", "Edmonton", "Ottawa", "Winnipeg", "Quebec City", "Hamilton", "Kitchener"],
    "Netherlands": ["Amsterdam", "Rotterdam", "The Hague", "Utrecht", "Eindhoven", "Tilburg", "Groningen", "Almere", "Breda", "Nijmegen"],
    "Italy": ["Rome", "Milan", "Naples", "Turin", "Palermo", "Genoa", "Bologna", "Florence", "Bari", "Catania"],
    "Spain": ["Madrid", "Barcelona", "Valencia", "Seville", "Zaragoza", "Malaga", "Murcia", "Palma", "Las Palmas", "Bilbao"],
    "South Korea": ["Seoul", "Busan", "Incheon", "Daegu", "Daejeon", "Gwangju", "Suwon", "Ulsan", "Changwon", "Goyang"],
    "Bangladesh": ["Dhaka", "Chittagong", "Khulna", "Rajshahi", "Sylhet", "Barisal", "Rangpur", "Comilla", "Narayanganj", "Gazipur"],
    "Myanmar": ["Yangon", "Mandalay", "Naypyidaw", "Mawlamyine", "Bago", "Pathein", "Monywa", "Meiktila", "Myitkyina", "Taunggyi"],
    "Sri Lanka": ["Colombo", "Kandy", "Galle", "Jaffna", "Negombo", "Anuradhapura", "Polonnaruwa", "Batticaloa", "Matara", "Trincomalee"],
    "Nepal": ["Kathmandu", "Pokhara", "Lalitpur", "Bharatpur", "Biratnagar", "Birgunj", "Dharan", "Butwal", "Hetauda", "Janakpur"],
    "Maldives": ["Malé", "Addu City", "Fuvahmulah"]
}

def get_gemini_response(prompt: str) -> str:
    """Get response from Google AI Studio API"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }
    data = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        return f"Error getting AI response: {str(e)}"

def calculate_distance_with_ai(origin: str, destination: str) -> Tuple[float, float]:
    """Calculate distance and duration using Google AI Studio API"""
    prompt = f"""
    Calculate the approximate driving distance and duration between {origin} and {destination}.
    
    Please provide:
    1. Distance in kilometers (just the number)
    2. Driving duration in hours (just the number)
    
    Format your response as:
    Distance: [number] km
    Duration: [number] hours
    
    Use your knowledge of geography and typical driving speeds to provide realistic estimates.
    For Indian cities, consider average speeds of 50-60 km/h for intercity travel.
    """
    
    try:
        response = get_gemini_response(prompt)
        
        # Parse the response to extract distance and duration
        distance_km = 0
        duration_hours = 0
        
        lines = response.split('\n')
        for line in lines:
            if 'Distance:' in line:
                try:
                    distance_km = float(line.split('Distance:')[1].split('km')[0].strip())
                except:
                    pass
            elif 'Duration:' in line:
                try:
                    duration_hours = float(line.split('Duration:')[1].split('hours')[0].strip())
                except:
                    pass
        
        # Fallback calculation if parsing fails
        if distance_km == 0 or duration_hours == 0:
            # Use a simple fallback based on city names
            distance_km = estimate_distance_fallback(origin, destination)
            duration_hours = distance_km / 55  # Average speed of 55 km/h
            
        return distance_km, duration_hours
        
    except Exception as e:
        st.error(f"Error calculating distance with AI: {str(e)}")
        # Fallback to simple estimation
        distance_km = estimate_distance_fallback(origin, destination)
        duration_hours = distance_km / 55
        return distance_km, duration_hours

def estimate_distance_fallback(origin: str, destination: str) -> float:
    """Simple fallback distance estimation"""
    # Basic distance estimates between major Indian cities
    city_distances = {
        ("Delhi", "Mumbai"): 1400,
        ("Delhi", "Chennai"): 2200,
        ("Delhi", "Kolkata"): 1500,
        ("Mumbai", "Chennai"): 1300,
        ("Mumbai", "Kolkata"): 2000,
        ("Chennai", "Kolkata"): 1650,
        ("Delhi", "Bangalore"): 2100,
        ("Mumbai", "Bangalore"): 980,
        ("Chennai", "Bangalore"): 350,
        ("Kolkata", "Bangalore"): 1900,
    }
    
    # Check if we have a direct mapping
    key = (origin, destination)
    reverse_key = (destination, origin)
    
    if key in city_distances:
        return city_distances[key]
    elif reverse_key in city_distances:
        return city_distances[reverse_key]
    else:
        # Default estimate based on city names
        return 1000  # Default 1000 km

def get_optimal_port(origin_city: str, destination_country: str, weight: float) -> Dict:
    """Find optimal port for shipment"""
    port_analysis = {}
    
    for port_city, info in WORLDREF_HUBS.items():
        if port_city == "Delhi":
            continue
            
        distance, duration = calculate_distance_with_ai(origin_city, port_city)
        weight_factor = get_weight_factor(weight)
        domestic_cost = distance * PRICING_CONFIG["base_rate_per_km_per_ton"] * weight_factor * weight
        port_charges = PRICING_CONFIG["port_charges"].get(port_city, 9000)
        
        # Check if port specializes in destination region
        specialization_bonus = 0.9 if destination_country in info["specialization"] else 1.0
        
        total_cost = (domestic_cost + port_charges) * specialization_bonus
        
        port_analysis[port_city] = {
            "distance_km": distance,
            "duration_hours": duration,
            "domestic_cost": domestic_cost,
            "port_charges": port_charges,
            "total_cost": total_cost,
            "specialization": info["specialization"],
            "suitable": destination_country in info["specialization"]
        }
    
    optimal_port = min(port_analysis.keys(), key=lambda x: port_analysis[x]["total_cost"])
    return {
        "port": optimal_port,
        "analysis": port_analysis
    }

def get_weight_factor(weight: float) -> float:
    """Get weight factor for pricing"""
    if weight < 1:
        return PRICING_CONFIG["weight_factors"]["< 1 ton"]
    elif weight <= 5:
        return PRICING_CONFIG["weight_factors"]["1-5 tons"]
    elif weight <= 10:
        return PRICING_CONFIG["weight_factors"]["5-10 tons"]
    else:
        return PRICING_CONFIG["weight_factors"]["> 10 tons"]

def calculate_total_cost(origin: str, destination: str, weight: float, 
                        commodity: str, service_level: str) -> Dict:
    """Calculate total shipping cost"""
    port_result = get_optimal_port(origin, destination, weight)
    optimal_port = port_result["port"]
    port_analysis = port_result["analysis"]
    
    domestic_cost = port_analysis[optimal_port]["domestic_cost"]
    port_charges = port_analysis[optimal_port]["port_charges"]
    
    commodity_factor = PRICING_CONFIG["commodity_factors"].get(commodity, 1.0)
    base_international_rate = 2000  # USD per ton
    international_cost = base_international_rate * weight * commodity_factor
    
    partner_rate = PRICING_CONFIG["partner_rates"][service_level]
    partner_cost = partner_rate * weight
    
    insurance_cost = (domestic_cost + international_cost) * 0.02
    documentation_cost = 5000
    
    total_cost = (domestic_cost + port_charges + international_cost + 
                  partner_cost + insurance_cost + documentation_cost)
    
    return {
        "total_cost": total_cost,
        "breakdown": {
            "domestic_transport": domestic_cost,
            "port_charges": port_charges,
            "international_shipping": international_cost,
            "partner_service": partner_cost,
            "insurance": insurance_cost,
            "documentation": documentation_cost
        },
        "optimal_port": optimal_port,
        "port_analysis": port_analysis,
        "distance_km": port_analysis[optimal_port]["distance_km"],
        "duration_hours": port_analysis[optimal_port]["duration_hours"]
    }

def generate_cost_explanation(cost_data: Dict, origin: str, destination: str, 
                            weight: float, commodity: str, service_level: str) -> str:
    """Generate AI-powered cost explanation"""
    prompt = f"""
    As a logistics expert for WorldRef, explain the following transport cost calculation:
    
    Origin: {origin}
    Destination: {destination}
    Weight: {weight} tons
    Commodity: {commodity}
    Service Level: {service_level}
    
    Selected Port: {cost_data['optimal_port']}
    Distance to Port: {cost_data['distance_km']:.0f} km
    Total Cost: ₹{cost_data['total_cost']:,.0f}
    
    Cost Breakdown:
    - Domestic Transport: ₹{cost_data['breakdown']['domestic_transport']:,.0f}
    - Port Charges: ₹{cost_data['breakdown']['port_charges']:,.0f}
    - International Shipping: ₹{cost_data['breakdown']['international_shipping']:,.0f}
    - Partner Service: ₹{cost_data['breakdown']['partner_service']:,.0f}
    - Insurance: ₹{cost_data['breakdown']['insurance']:,.0f}
    - Documentation: ₹{cost_data['breakdown']['documentation']:,.0f}
    
    Available ports and their analysis:
    {json.dumps(cost_data['port_analysis'], indent=2)}
    
    Please provide:
    1. Why this specific port was selected
    2. Explanation of each cost component
    3. Alternative options and their trade-offs
    4. Any recommendations for cost optimization
    
    Keep the explanation professional but easy to understand.
    """
    
    return get_gemini_response(prompt)

def main():
    st.set_page_config(
        page_title="WorldRef Transport Cost Estimator",
        page_icon="🚚",
        layout="wide"
    )
    
    st.title("🚚 WorldRef Transport Cost Estimator")
    st.markdown("Real-time transport cost calculation with AI-powered optimization")
    
    st.sidebar.header("📋 Shipment Details")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        # Origin city input with both dropdown and text input
        origin_option = st.radio("Origin City Selection", ["Select from list", "Enter manually"])
        if origin_option == "Select from list":
            origin = st.selectbox("Origin City", INDIAN_CITIES, index=0)
        else:
            origin = st.text_input("Enter Origin City", value="Delhi")
        
        weight = st.number_input("Weight (tons)", min_value=0.1, max_value=100.0, value=1.0, step=0.1)
        service_level = st.selectbox("Service Level", ["Express", "Standard", "Economy"], index=1)
    
    with col2:
        # Destination country and city selection
        destination_country = st.selectbox("Destination Country", DESTINATION_COUNTRIES, index=0)
        
        # Destination city input
        dest_option = st.radio("Destination City Selection", ["Select from list", "Enter manually"])
        if dest_option == "Select from list":
            if destination_country in DESTINATION_CITIES:
                destination_city = st.selectbox("Destination City", DESTINATION_CITIES[destination_country], index=0)
            else:
                destination_city = st.text_input("Enter Destination City", value="")
        else:
            destination_city = st.text_input("Enter Destination City", value="")
        
        commodity = st.selectbox("Commodity Type", list(PRICING_CONFIG["commodity_factors"].keys()), index=0)
        preferred_date = st.date_input("Preferred Delivery Date", datetime.now() + timedelta(days=7))
    
    if st.sidebar.button("💰 Calculate Cost", type="primary"):
        if not destination_city:
            st.error("Please enter a destination city")
            return
            
        with st.spinner("Calculating optimal route and costs..."):
            cost_data = calculate_total_cost(origin, destination_country, weight, commodity, service_level)
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.metric(
                    "Total Cost",
                    f"₹{cost_data['total_cost']:,.0f}",
                    f"~${cost_data['total_cost']/83:,.0f} USD"
                )
            
            with col2:
                st.metric(
                    "Optimal Port",
                    cost_data['optimal_port'],
                    f"{cost_data['distance_km']:.0f} km"
                )
            
            with col3:
                st.metric(
                    "Estimated Transit",
                    f"{cost_data['duration_hours']:.1f} hours",
                    "to port"
                )
            
            st.subheader("📊 Cost Breakdown")
            
            breakdown_data = cost_data['breakdown']
            
            fig = px.pie(
                values=list(breakdown_data.values()),
                names=list(breakdown_data.keys()),
                title="Cost Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("💰 Detailed Cost Breakdown")
                breakdown_df = pd.DataFrame([
                    {"Component": k.replace("_", " ").title(), "Cost (₹)": f"{v:,.0f}"}
                    for k, v in breakdown_data.items()
                ])
                st.dataframe(breakdown_df, use_container_width=True)
            
            with col2:
                st.subheader("🚢 Port Analysis")
                port_df = pd.DataFrame([
                    {
                        "Port": port,
                        "Distance (km)": f"{data['distance_km']:.0f}",
                        "Total Cost (₹)": f"{data['total_cost']:,.0f}",
                        "Suitable": "✅" if data['suitable'] else "❌"
                    }
                    for port, data in cost_data['port_analysis'].items()
                ])
                st.dataframe(port_df, use_container_width=True)
            
            st.subheader("🤖 AI Cost Analysis")
            
            with st.spinner("Generating detailed explanation..."):
                explanation = generate_cost_explanation(
                    cost_data, origin, f"{destination_city}, {destination_country}", 
                    weight, commodity, service_level
                )
            
            st.markdown(explanation)
            
            st.subheader("📈 Additional Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"""
                **Route Optimization:**
                - Selected port: {cost_data['optimal_port']}
                - Distance: {cost_data['distance_km']:.0f} km
                - Estimated delivery: {preferred_date + timedelta(days=7)}
                """)
            
            with col2:
                st.warning(f"""
                **Cost Factors:**
                - Weight factor: {get_weight_factor(weight)}x
                - Commodity factor: {PRICING_CONFIG['commodity_factors'][commodity]}x
                - Service level: {service_level}
                """)
            
            st.subheader("📄 Export Options")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📋 Generate Quote"):
                    quote_text = f"""
WORLDREF TRANSPORT QUOTE

From: {origin}
To: {destination_city}, {destination_country}
Weight: {weight} tons
Commodity: {commodity}
Service: {service_level}

Via: {cost_data['optimal_port']} Port
Total Cost: ₹{cost_data['total_cost']:,.0f}

Valid until: {datetime.now() + timedelta(days=7)}
                    """
                    st.text_area("Quote", quote_text, height=200)
            
            with col2:
                if st.button("📊 Download Report"):
                    st.success("Report generation feature coming soon!")
            
            with col3:
                if st.button("🔄 Book Shipment"):
                    st.success("Booking integration coming soon!")

if __name__ == "__main__":
    main()
