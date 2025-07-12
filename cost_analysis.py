import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional
import math

GEMINI_API_KEY = "AIzaSyB46mW-7p4MIrKSe-oudQLpjxWli6XjVpE"
GEMINI_MODEL = "gemini-2.0-flash-001"

WORLDREF_HUBS = {
    "Mumbai": {"lat": 19.0760, "lng": 72.8777, "port": "JNPT", "specialization": "Middle East, Europe"},
    "Chennai": {"lat": 13.0827, "lng": 80.2707, "port": "Chennai Port", "specialization": "Southeast Asia"},
    "Kolkata": {"lat": 22.5726, "lng": 88.3639, "port": "Kolkata Port", "specialization": "Bangladesh, Myanmar"},
    "Delhi": {"lat": 28.6139, "lng": 77.2090, "port": "Main Hub", "specialization": "Inland Hub"}
}

PRICING_CONFIG = {
    "base_rate_per_km_per_ton": 15,  # Road transportation within India
    "rail_rate_per_km_per_ton": 8,   # Rail transportation
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
        "Electronics": 1.2,
        "Textiles": 1.1,
        "Machinery": 1.3,
        "Chemicals": 1.4,
        "Food Products": 1.2
    },
    "port_charges": {
        "Mumbai": 10000,
        "Chennai": 9000,
        "Kolkata": 8500
    },
    "international_rates": {
        "Air": {
            "per_kg": 250,       # ₹ per kg
            "per_cbm": 8000      # ₹ per CBM
        },
        "Sea": {
            "per_kg": 45,        # ₹ per kg
            "per_cbm": 1200      # ₹ per CBM
        },
        "Road": {
            "per_km_per_ton": 18 # ₹ per km per ton
        }
    },
    "consolidation_factors": {
        "handling_fee_per_supplier": 2000,
        "consolidation_discount": 0.95, 
        "volume_efficiency_factor": 0.85,
        "min_consolidation_weight": 0.1,
        "warehouse_fee_per_day": 500,
        "consolidation_time_days": 2
    },
    "international_mode_surcharges": {
        "Air": 1.3,
        "Sea": 1.0,
        "Road": 1.1
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

INTERNATIONAL_MODES = ["Air", "Sea", "Road"] 

class PackingItem:
    def __init__(self, description: str, quantity: int, weight_per_unit: float, 
                 volume_per_unit: float, commodity_type: str, value: float,
                 location: str):
        self.description = description
        self.quantity = quantity
        self.weight_per_unit = weight_per_unit
        self.volume_per_unit = volume_per_unit
        self.commodity_type = commodity_type
        self.value = value
        self.location = location
        self.total_weight = quantity * weight_per_unit
        self.total_volume = quantity * volume_per_unit
        # Add tracking information
        self.transport_stages = []
        self.cost_breakdown = {}
        
    def add_transport_stage(self, stage: str, cost: float, details: str):
        """Add a transport stage with cost and details"""
        self.transport_stages.append({
            "stage": stage,
            "cost": cost,
            "details": details
        })
        if stage not in self.cost_breakdown:
            self.cost_breakdown[stage] = 0
        self.cost_breakdown[stage] += cost
        
    def get_total_cost(self) -> float:
        """Get total cost for this item"""
        return sum(stage["cost"] for stage in self.transport_stages)

class Supplier:
    def __init__(self, name: str, contact: str):
        self.name = name
        self.contact = contact
        self.packing_list: List[PackingItem] = []
        self.total_weight = 0
        self.total_volume = 0
        self.total_value = 0
        self.locations = set()      
    def add_packing_item(self, item: PackingItem):
        self.packing_list.append(item)
        self.total_weight += item.total_weight
        self.total_volume += item.total_volume
        self.total_value += item.value * item.quantity
        self.locations.add(item.location)    
    def get_dominant_commodity(self) -> str:
        if not self.packing_list:
            return "General Goods"    
        commodity_values = {}
        for item in self.packing_list:
            commodity = item.commodity_type
            commodity_values[commodity] = commodity_values.get(commodity, 0) + (item.value * item.quantity)        
        return max(commodity_values, key=commodity_values.get)   
    def get_locations_summary(self) -> str:
        return ", ".join(self.locations) if self.locations else "N/A"  

def get_gemini_response(prompt: str) -> str:
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
        if distance_km == 0 or duration_hours == 0:
            distance_km = estimate_distance_fallback(origin, destination)
            duration_hours = distance_km / 55  # Average speed of 55 km/h         
        return distance_km, duration_hours 
    except Exception as e:
        st.error(f"Error calculating distance with AI: {str(e)}")
        distance_km = estimate_distance_fallback(origin, destination)
        duration_hours = distance_km / 55
        return distance_km, duration_hours

def estimate_distance_fallback(origin: str, destination: str) -> float:
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
    key = (origin, destination)
    reverse_key = (destination, origin)
    if key in city_distances:
        return city_distances[key]
    elif reverse_key in city_distances:
        return city_distances[reverse_key]
    else:
        return 1000

def analyze_consolidation_benefits(suppliers: List[Supplier], consolidation_hub: str) -> Dict:
    total_weight = 0
    total_volume = 0
    total_value = 0
    commodity_types = set()    
    for supplier in suppliers:
        for item in supplier.packing_list:
            total_weight += item.total_weight
            total_volume += item.total_volume
            total_value += item.value * item.quantity
            commodity_types.add(item.commodity_type)   
    incompatible_pairs = [
        ("Hazardous Materials", "Food Products"),
        ("Hazardous Materials", "Perishable Goods"),
        ("Chemicals", "Food Products"),
        ("Chemicals", "Perishable Goods")
    ]      
    is_compatible = True
    for pair in incompatible_pairs:
        if pair[0] in commodity_types and pair[1] in commodity_types:
            is_compatible = False
            break               
    consolidation_feasible = (
        total_weight >= PRICING_CONFIG["consolidation_factors"]["min_consolidation_weight"] and
        is_compatible
    )      
    if consolidation_feasible:
        optimized_volume = total_volume * PRICING_CONFIG["consolidation_factors"]["volume_efficiency_factor"]
        handling_fees = len(suppliers) * PRICING_CONFIG["consolidation_factors"]["handling_fee_per_supplier"]
        warehouse_fees = (PRICING_CONFIG["consolidation_factors"]["consolidation_time_days"] * 
                         PRICING_CONFIG["consolidation_factors"]["warehouse_fee_per_day"])       
        consolidation_costs = handling_fees + warehouse_fees
        shipping_discount = PRICING_CONFIG["consolidation_factors"]["consolidation_discount"]     
        return {
            "feasible": True,
            "total_weight": total_weight,
            "total_volume": total_volume,
            "optimized_volume": optimized_volume,
            "total_value": total_value,
            "consolidation_costs": consolidation_costs,
            "shipping_discount": shipping_discount,
            "commodity_compatibility": is_compatible,
            "incompatible_reason": None if is_compatible else "Incompatible commodities detected",
            "consolidation_time_days": PRICING_CONFIG["consolidation_factors"]["consolidation_time_days"]
        }
    else:
        return {
            "feasible": False,
            "total_weight": total_weight,
            "total_volume": total_volume,
            "optimized_volume": total_volume,
            "total_value": total_value,
            "consolidation_costs": 0,
            "shipping_discount": 1.0,
            "commodity_compatibility": is_compatible,
            "incompatible_reason": "Minimum weight not met" if total_weight < PRICING_CONFIG["consolidation_factors"]["min_consolidation_weight"] else "Incompatible commodities",
            "consolidation_time_days": 0
        }

def get_optimal_port(suppliers: List[Supplier], destination_country: str, 
                    consolidation_analysis: Dict, international_mode: str) -> Dict:
    port_analysis = {}
    
    # Determine preferred ports based on mode
    preferred_ports = {
        "Sea": ["Mumbai", "Chennai", "Kolkata"],
        "Air": ["Delhi", "Mumbai", "Chennai"],
        "Road": ["Delhi"]
    }
    port_candidates = preferred_ports.get(international_mode, list(WORLDREF_HUBS.keys()))
    
    for port_city in port_candidates:
        if port_city not in WORLDREF_HUBS:
            continue
            
        info = WORLDREF_HUBS[port_city]
        total_domestic_cost = 0
        max_distance = 0
        
        # Track consolidation point
        consolidation_point = port_city
        if port_city == "Delhi" and international_mode == "Sea":
            consolidation_point = "Delhi (via Rail to Mumbai/Chennai)"
        
        for supplier in suppliers:
            for item in supplier.packing_list:
                distance, duration = calculate_distance_with_ai(item.location, port_city)
                weight_factor = get_weight_factor(item.total_weight)
                base_rate = PRICING_CONFIG["base_rate_per_km_per_ton"]
                
                # Add rail cost for Delhi to port
                transport_mode = "Road"
                if item.location == "Delhi" and port_city != "Delhi" and international_mode == "Sea":
                    base_rate = PRICING_CONFIG["rail_rate_per_km_per_ton"]
                    transport_mode = "Rail"
                
                item_cost = distance * base_rate * weight_factor * item.total_weight
                commodity_factor = PRICING_CONFIG["commodity_factors"].get(item.commodity_type, 1.0)
                item_cost *= commodity_factor
                
                # Add stage information to item
                item.add_transport_stage(
                    stage=f"Domestic {transport_mode}",
                    cost=item_cost,
                    details=f"{item.location} to {port_city} hub ({distance} km)"
                )
                
                total_domestic_cost += item_cost
                max_distance = max(max_distance, distance)
        
        if consolidation_analysis["feasible"]:
            total_domestic_cost += consolidation_analysis["consolidation_costs"]
        
        port_charges = PRICING_CONFIG["port_charges"].get(port_city, 9000)
        specialization_bonus = 0.9 if destination_country in info["specialization"] else 1.0
        
        total_cost = total_domestic_cost + port_charges * specialization_bonus
        
        port_analysis[port_city] = {
            "total_domestic_cost": total_domestic_cost,
            "max_distance_km": max_distance,
            "port_charges": port_charges,
            "total_cost": total_cost,
            "specialization": info["specialization"],
            "suitable": destination_country in info["specialization"],
            "consolidation_point": consolidation_point
        }
    
    if not port_analysis:  # Fallback if no ports found
        port_city = "Mumbai"
        info = WORLDREF_HUBS[port_city]
        port_analysis[port_city] = {
            "total_domestic_cost": 0,
            "max_distance_km": 0,
            "port_charges": PRICING_CONFIG["port_charges"].get(port_city, 9000),
            "total_cost": 0,
            "specialization": info["specialization"],
            "suitable": destination_country in info["specialization"],
            "consolidation_point": port_city
        }
    
    optimal_port = min(port_analysis.keys(), key=lambda x: port_analysis[x]["total_cost"])
    
    return {
        "port": optimal_port,
        "analysis": port_analysis
    }

def get_weight_factor(weight: float) -> float:
    if weight < 1:
        return PRICING_CONFIG["weight_factors"]["< 1 ton"]
    elif weight <= 5:
        return PRICING_CONFIG["weight_factors"]["1-5 tons"]
    elif weight <= 10:
        return PRICING_CONFIG["weight_factors"]["5-10 tons"]
    else:
        return PRICING_CONFIG["weight_factors"]["> 10 tons"]

def calculate_international_shipping_cost(total_weight: float, total_volume: float, 
                                        destination_country: str, 
                                        consolidation_analysis: Dict,
                                        international_mode: str,
                                        suppliers: List[Supplier]) -> float:
    mode_rates = PRICING_CONFIG["international_rates"][international_mode]
    surcharge = PRICING_CONFIG["international_mode_surcharges"].get(international_mode, 1.0)
    
    if international_mode == "Road":
        avg_road_distance = {
            "Bangladesh": 500,
            "Nepal": 700,
            "Bhutan": 1000,
            "Myanmar": 1500,
            "Sri Lanka": 2500 
        }
        distance = avg_road_distance.get(destination_country, 2000)
        cost = distance * mode_rates["per_km_per_ton"] * total_weight
        
        # Add international stage to each item
        for supplier in suppliers:
            for item in supplier.packing_list:
                item.add_transport_stage(
                    stage="International Road",
                    cost=item.total_weight * distance * mode_rates["per_km_per_ton"] * surcharge,
                    details=f"To {destination_country} ({distance} km)"
                )
    else:
        weight_cost = total_weight * 1000 * mode_rates["per_kg"]  # Convert tons to kg
        volume_cost = consolidation_analysis["optimized_volume"] * mode_rates["per_cbm"]
        cost = max(weight_cost, volume_cost)
        
        # Add international stage to each item
        for supplier in suppliers:
            for item in supplier.packing_list:
                # Calculate proportional cost
                weight_ratio = item.total_weight / total_weight if total_weight > 0 else 0
                volume_ratio = item.total_volume / consolidation_analysis["total_volume"] if consolidation_analysis["total_volume"] > 0 else 0
                item_cost = max(
                    weight_ratio * weight_cost,
                    volume_ratio * volume_cost
                ) * surcharge
                
                item.add_transport_stage(
                    stage=f"International {international_mode}",
                    cost=item_cost,
                    details=f"To {destination_country}"
                )
    
    return cost * consolidation_analysis["shipping_discount"]

def calculate_multi_supplier_cost(suppliers: List[Supplier], destination_country: str, 
                                destination_port: str, international_mode: str) -> Dict:
    # Clear any existing transport stages
    for supplier in suppliers:
        for item in supplier.packing_list:
            item.transport_stages = []
            item.cost_breakdown = {}
    
    consolidation_analysis = analyze_consolidation_benefits(suppliers, "Mumbai")
    port_result = get_optimal_port(suppliers, destination_country, consolidation_analysis, international_mode)
    optimal_port = port_result["port"]
    port_analysis = port_result["analysis"]
    
    domestic_cost = port_analysis[optimal_port]["total_domestic_cost"]
    port_charges = port_analysis[optimal_port]["port_charges"]
    
    # Add port charges to each item proportionally
    total_weight = consolidation_analysis["total_weight"]
    for supplier in suppliers:
        for item in supplier.packing_list:
            weight_ratio = item.total_weight / total_weight if total_weight > 0 else 0
            item.add_transport_stage(
                stage="Port Handling",
                cost=weight_ratio * port_charges,
                details=f"{optimal_port} port charges"
            )
    
    international_cost = calculate_international_shipping_cost(
        consolidation_analysis["total_weight"],
        consolidation_analysis["total_volume"],
        destination_country,
        consolidation_analysis,
        international_mode,
        suppliers
    )
    
    insurance_cost = consolidation_analysis["total_value"] * 0.02
    documentation_cost = 5000
    
    # Add insurance and documentation proportionally
    total_value = consolidation_analysis["total_value"]
    for supplier in suppliers:
        for item in supplier.packing_list:
            value_ratio = (item.value * item.quantity) / total_value if total_value > 0 else 0
            item.add_transport_stage(
                stage="Insurance",
                cost=value_ratio * insurance_cost,
                details="Cargo insurance"
            )
            item.add_transport_stage(
                stage="Documentation",
                cost=value_ratio * documentation_cost,
                details="Shipping documentation"
            )
    
    # Add consolidation costs if applicable
    if consolidation_analysis["feasible"]:
        handling_fee = PRICING_CONFIG["consolidation_factors"]["handling_fee_per_supplier"]
        warehouse_fee = PRICING_CONFIG["consolidation_factors"]["warehouse_fee_per_day"]
        consolidation_time = PRICING_CONFIG["consolidation_factors"]["consolidation_time_days"]
        consolidation_cost_per_supplier = handling_fee + (warehouse_fee * consolidation_time)
        
        for supplier in suppliers:
            for item in supplier.packing_list:
                item.add_transport_stage(
                    stage="Consolidation",
                    cost=consolidation_cost_per_supplier / len(supplier.packing_list),
                    details=f"Handling & warehousing at {optimal_port}"
                )
    
    total_cost = domestic_cost + port_charges + international_cost + insurance_cost + documentation_cost
    
    return {
        "total_cost": total_cost,
        "breakdown": {
            "domestic_transport": domestic_cost,
            "port_charges": port_charges,
            "international_shipping": international_cost,
            "insurance": insurance_cost,
            "documentation": documentation_cost,
            "consolidation_costs": consolidation_analysis.get("consolidation_costs", 0)
        },
        "consolidation_analysis": consolidation_analysis,
        "optimal_port": optimal_port,
        "port_analysis": port_analysis,
        "suppliers_summary": {
            "count": len(suppliers),
            "total_weight": consolidation_analysis["total_weight"],
            "total_volume": consolidation_analysis["total_volume"],
            "total_value": consolidation_analysis["total_value"]
        },
        "international_mode": international_mode,
        "consolidation_point": port_analysis[optimal_port]["consolidation_point"]
    }

def main():
    st.set_page_config(
        page_title="WorldRef Multi-Supplier Consolidation Estimator",
        page_icon="📦",
        layout="wide"
    )   
    st.title("📦 WorldRef Multi-Supplier Consolidation Estimator")
    st.markdown("Advanced transport cost calculation with multi-supplier consolidation and optimization")  
    if 'suppliers' not in st.session_state:
        st.session_state.suppliers = []   
    st.sidebar.header("📋 Shipment Details")
    destination_country = st.sidebar.selectbox("Destination Country", DESTINATION_COUNTRIES, index=0)
    international_mode = st.sidebar.selectbox("International Transport Mode", INTERNATIONAL_MODES, index=1)
    destination_port = st.sidebar.text_input("Destination Port", value="Main Port")   
    tab1, tab2, tab3 = st.tabs(["📦 Supplier Management", "💰 Cost Analysis", "📊 Consolidation Report"])   
    with tab1:
        st.header("Supplier and Packing List Management")
        st.subheader("Add New Supplier")       
        col1, col2 = st.columns(2)       
        with col1:
            supplier_name = st.text_input("Supplier Name")
        with col2:
            supplier_contact = st.text_input("Contact Information")    
        if st.button("Add Supplier"):
            if supplier_name and supplier_name.strip():
                new_supplier = Supplier(
                    supplier_name.strip(), 
                    supplier_contact
                )
                st.session_state.suppliers.append(new_supplier)
                st.success(f"Supplier {supplier_name} added successfully!")
                st.rerun()
            else:
                st.error("Please fill in supplier name")      
        if st.session_state.suppliers:
            st.subheader("Existing Suppliers")
            for i, supplier in enumerate(st.session_state.suppliers):
                with st.expander(f"📍 {supplier.name}"):
                    col1, col2 = st.columns([3, 1])                   
                    with col1:
                        st.write(f"**Contact:** {supplier.contact}")
                        st.write(f"**Total Weight:** {supplier.total_weight:.2f} tons")
                        st.write(f"**Total Volume:** {supplier.total_volume:.2f} CBM")
                        st.write(f"**Total Value:** ${supplier.total_value:,.2f}")
                        st.write(f"**Locations:** {supplier.get_locations_summary()}")
                    with col2:
                        if st.button(f"Remove {supplier.name}", key=f"remove_{i}"):
                            st.session_state.suppliers.pop(i)
                            st.rerun()                   
                    st.write("**Add Packing Items:**")                   
                    with st.form(key=f"item_form_{i}"):
                        pcol1, pcol2, pcol3, pcol4 = st.columns(4)                       
                        with pcol1:
                            item_desc = st.text_input("Item Description", key=f"desc_{i}")
                            item_qty = st.number_input("Quantity", min_value=1, value=1, key=f"qty_{i}")
                        with pcol2:
                            item_weight = st.number_input("Weight per Unit (kg)", min_value=0.01, value=1.0, step=0.01, key=f"weight_{i}")
                            item_volume = st.number_input("Volume per Unit (CBM)", min_value=0.001, value=0.1, step=0.001, key=f"volume_{i}")
                        with pcol3:
                            item_value = st.number_input("Value per Unit ($)", min_value=0.01, value=10.0, step=0.01, key=f"value_{i}")
                            item_commodity = st.selectbox("Commodity Type", 
                                                        list(PRICING_CONFIG["commodity_factors"].keys()),
                                                        key=f"commodity_{i}")
                        with pcol4:
                            item_location = st.selectbox("Location", INDIAN_CITIES, key=f"location_{i}")                           
                        add_item_button = st.form_submit_button(f"Add Item to {supplier.name}")                       
                        if add_item_button:
                            if item_desc and item_desc.strip():
                                new_item = PackingItem(
                                    item_desc, item_qty, item_weight/1000,  # Convert kg to tons
                                    item_volume, item_commodity, item_value,
                                    item_location
                                )
                                supplier.add_packing_item(new_item)
                                st.success(f"Item '{item_desc}' added to {supplier.name}'s packing list!")
                                st.rerun()
                            else:
                                st.error("Please enter item description")                   
                    if supplier.packing_list:
                        st.write("**Current Packing List:**")
                        packing_data = []
                        for item in supplier.packing_list:
                            packing_data.append({
                                "Description": item.description,
                                "Quantity": item.quantity,
                                "Unit Weight (kg)": f"{item.weight_per_unit * 1000:.2f}",
                                "Unit Volume (CBM)": f"{item.volume_per_unit:.3f}",
                                "Commodity": item.commodity_type,
                                "Unit Value ($)": f"{item.value:.2f}",
                                "Total Weight (kg)": f"{item.total_weight * 1000:.2f}",
                                "Total Volume (CBM)": f"{item.total_volume:.3f}",
                                "Total Value ($)": f"{item.value * item.quantity:.2f}",
                                "Location": item.location
                            })
                        packing_df = pd.DataFrame(packing_data)
                        st.dataframe(packing_df, use_container_width=True)                       
                        if st.button(f"Clear Packing List", key=f"clear_{i}"):
                            supplier.packing_list = []
                            supplier.total_weight = 0
                            supplier.total_volume = 0
                            supplier.total_value = 0
                            supplier.locations = set()
                            st.success("Packing list cleared!")
                            st.rerun()
                    else:
                        st.info("No items in packing list. Add items above.")   
    with tab2:
        st.header("Cost Analysis")        
        if len(st.session_state.suppliers) >= 1:
            if st.button("🔍 Analyze Consolidation & Calculate Costs", type="primary"):
                with st.spinner("Analyzing consolidation options and calculating costs..."):
                    cost_data = calculate_multi_supplier_cost(
                        st.session_state.suppliers, destination_country, 
                        destination_port, international_mode
                    )                 
                    col1, col2, col3, col4 = st.columns(4)                  
                    with col1:
                        st.metric("Total Cost", f"₹{cost_data['total_cost']:,.0f}", 
                                f"~${cost_data['total_cost']/83:,.0f} USD")                    
                    with col2:
                        st.metric("Total Weight", f"{cost_data['suppliers_summary']['total_weight']:.2f} tons")                   
                    with col3:
                        st.metric("Total Volume", f"{cost_data['suppliers_summary']['total_volume']:.2f} CBM")              
                    with col4:
                        st.metric("Transport Mode", f"Domestic: Road\nInternational: {cost_data['international_mode']}")        
                    st.subheader("📊 Cost Breakdown")                    
                    breakdown_data = cost_data['breakdown']
                    fig = px.pie(
                        values=list(breakdown_data.values()),
                        names=[name.replace("_", " ").title() for name in breakdown_data.keys()],
                        title="Cost Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)                   
                    breakdown_df = pd.DataFrame([
                        {"Component": k.replace("_", " ").title(), "Cost (₹)": f"{v:,.0f}"}
                        for k, v in breakdown_data.items()
                    ])
                    st.dataframe(breakdown_df, use_container_width=True)                   
                    st.subheader("🔄 Consolidation Analysis")                  
                    cons_analysis = cost_data['consolidation_analysis'] 
                    consolidation_point = cost_data["consolidation_point"]                  
                    if cons_analysis['feasible']:
                        st.success(f"✅ Consolidation is feasible at {consolidation_point}!")                        
                        col1, col2 = st.columns(2)                        
                        with col1:
                            st.info(f"""
                            **Consolidation Benefits:**
                            - Volume optimization: {cons_analysis['optimized_volume']:.2f} CBM 
                              (vs {cons_analysis['total_volume']:.2f} CBM original)
                            - Shipping discount: {(1-cons_analysis['shipping_discount'])*100:.0f}%
                            - Consolidation time: {cons_analysis['consolidation_time_days']} days
                            """)                       
                        with col2:
                            st.warning(f"""
                            **Consolidation Costs:**
                            - Handling fees: ₹{cons_analysis['consolidation_costs']:,.0f}
                            - Compatible commodities: ✅
                            - Total suppliers: {cost_data['suppliers_summary']['count']}
                            """)
                    else:
                        st.error("❌ Consolidation not feasible")
                        st.write(f"**Reason:** {cons_analysis['incompatible_reason']}")                   
                    st.subheader("🚢 Port Analysis")
                    port_data = []
                    for port, data in cost_data['port_analysis'].items():
                        port_data.append({
                            "Port": port,
                            "Domestic Cost (₹)": f"{data['total_domestic_cost']:,.0f}",
                            "Port Charges (₹)": f"{data['port_charges']:,.0f}",
                            "Total Cost (₹)": f"{data['total_cost']:,.0f}",
                            "Specialization": data['specialization'],
                            "Suitable": "✅" if data['suitable'] else "❌"
                        })                    
                    port_df = pd.DataFrame(port_data)
                    st.dataframe(port_df, use_container_width=True)            
                    st.subheader("✈️ International Shipping Details")
                    intl_col1, intl_col2 = st.columns(2)
                    with intl_col1:
                        st.write(f"**Mode:** {cost_data['international_mode']}")
                        if cost_data['international_mode'] == "Road":
                            st.write("**Applicability:** Only for neighboring countries")
                        st.write(f"**Surcharge:** {PRICING_CONFIG['international_mode_surcharges'][cost_data['international_mode']]*100:.0f}%")
                    with intl_col2:
                        rates = PRICING_CONFIG["international_rates"][cost_data['international_mode']]
                        if cost_data['international_mode'] == "Road":
                            st.write(f"**Rate:** ₹{rates['per_km_per_ton']} per km per ton")
                        else:
                            st.write(f"**Air/Sea Rates:**")
                            st.write(f"- Per kg: ₹{rates['per_kg']}")
                            st.write(f"- Per CBM: ₹{rates['per_cbm']}") 
                    
                    # FIXED: Per-Item Cost Breakdown with unique keys
                    st.subheader("📦 Per-Item Cost Breakdown")
                    for s_idx, supplier in enumerate(st.session_state.suppliers):
                        with st.expander(f"📦 {supplier.name} - Item Costs"):
                            for i_idx, item in enumerate(supplier.packing_list):
                                st.markdown(f"**{item.description}** (Qty: {item.quantity})")
                                
                                # Create table for cost stages
                                stage_data = []
                                for stage in item.transport_stages:
                                    stage_data.append({
                                        "Stage": stage["stage"],
                                        "Cost (₹)": f"{stage['cost']:,.0f}",
                                        "Details": stage["details"]
                                    })
                                
                                stage_df = pd.DataFrame(stage_data)
                                st.dataframe(stage_df, hide_index=True)
                                
                                total_item_cost = item.get_total_cost()
                                st.markdown(f"**Total Item Cost: ₹{total_item_cost:,.0f}**")
                                
                                # Pie chart for cost distribution with unique key
                                if total_item_cost > 0:
                                    cost_dist = {}
                                    for stage in stage_data:
                                        cost_value = float(stage["Cost (₹)"].replace(",", ""))
                                        stage_key = f"{stage['Stage']}_{s_idx}_{i_idx}"
                                        cost_dist[stage_key] = cost_value
                                    
                                    fig = px.pie(
                                        names=list(cost_dist.keys()),
                                        values=list(cost_dist.values()),
                                        title=f"Cost Distribution for {item.description}"
                                    )
                                    st.plotly_chart(fig, use_container_width=True, 
                                                    key=f"pie_{s_idx}_{i_idx}")
        else:
            st.info("Please add at least one supplier to calculate costs.")   
    with tab3:
        st.header("📊 Consolidation Report")        
        if len(st.session_state.suppliers) >= 1:
            st.subheader("Supplier Summary")           
            summary_data = []
            for supplier in st.session_state.suppliers:
                summary_data.append({
                    "Supplier": supplier.name,
                    "Items": len(supplier.packing_list),
                    "Weight (tons)": f"{supplier.total_weight:.2f}",
                    "Volume (CBM)": f"{supplier.total_volume:.2f}",
                    "Value ($)": f"{supplier.total_value:,.2f}",
                    "Dominant Commodity": supplier.get_dominant_commodity(),
                    "Locations": supplier.get_locations_summary()
                })          
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            st.subheader("📋 Item Details")
            item_data = []
            for supplier in st.session_state.suppliers:
                for item in supplier.packing_list:
                    item_data.append({
                        "Supplier": supplier.name,
                        "Item": item.description,
                        "Quantity": item.quantity,
                        "Weight (kg)": f"{item.weight_per_unit * 1000:.2f}",
                        "Volume (CBM)": f"{item.volume_per_unit:.3f}",
                        "Commodity": item.commodity_type,
                        "Value ($)": f"{item.value:.2f}",
                        "Location": item.location
                    })
            item_df = pd.DataFrame(item_data)
            st.dataframe(item_df, use_container_width=True)           
            st.subheader("📋 Transport Logic & Assumptions")            
            st.markdown("""
            **Transportation Strategy:**
            - **Within India (Source to Port):** Road transport (Rail for Delhi to port for sea shipments)
            - **International Shipping:** User-selected mode (Air/Sea/Road)
            
            **Domestic Transport Logic:**
            - Base rate: ₹15 per km per ton (Road), ₹8 per km per ton (Rail)
            - Weight-based discounts for larger shipments
            - Commodity-specific surcharges
            
            **International Shipping Logic:**
            - **Air:** Charged per kg or per CBM (whichever is higher)
            - **Sea:** Charged per kg or per CBM (whichever is higher)
            - **Road:** Charged per km per ton (only for neighboring countries)
            
            **Consolidation Benefits:**
            - Volume optimization through better packing
            - Discounts on international shipping
            - Reduced per-unit handling costs
            
            **Port Selection:**
            - Based on specialization for destination region
            - Minimizing total domestic transport costs
            """)
        else:
            st.info("Add suppliers to view consolidation report.")

if __name__ == "__main__":
    main()
