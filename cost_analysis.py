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
    "partner_rates": {
        "Express": 220,
        "Standard": 150,
        "Economy": 100
    },
    "consolidation_factors": {
        "handling_fee_per_supplier": 2000,  # Fixed fee per supplier
        "consolidation_discount": 0.95,     # 5% discount for consolidated shipments
        "volume_efficiency_factor": 0.85,   # 15% volume efficiency gain
        "min_consolidation_weight": 2.0,    # Minimum weight for consolidation benefits
        "warehouse_fee_per_day": 500,       # Daily warehouse storage fee
        "consolidation_time_days": 2        # Time needed for consolidation
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

class PackingItem:
    def __init__(self, description: str, quantity: int, weight_per_unit: float, 
                 volume_per_unit: float, commodity_type: str, value: float):
        self.description = description
        self.quantity = quantity
        self.weight_per_unit = weight_per_unit
        self.volume_per_unit = volume_per_unit
        self.commodity_type = commodity_type
        self.value = value
        self.total_weight = quantity * weight_per_unit
        self.total_volume = quantity * volume_per_unit

class Supplier:
    def __init__(self, name: str, location: str, contact: str):
        self.name = name
        self.location = location
        self.contact = contact
        self.packing_list: List[PackingItem] = []
        self.total_weight = 0
        self.total_volume = 0
        self.total_value = 0
    
    def add_packing_item(self, item: PackingItem):
        self.packing_list.append(item)
        self.total_weight += item.total_weight
        self.total_volume += item.total_volume
        self.total_value += item.value * item.quantity
    
    def get_dominant_commodity(self) -> str:
        """Get the commodity type with highest value"""
        if not self.packing_list:
            return "General Goods"
        
        commodity_values = {}
        for item in self.packing_list:
            commodity = item.commodity_type
            commodity_values[commodity] = commodity_values.get(commodity, 0) + (item.value * item.quantity)
        
        return max(commodity_values, key=commodity_values.get)

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
        return 1000  # Default 1000 km

def analyze_consolidation_benefits(suppliers: List[Supplier], consolidation_hub: str) -> Dict:
    """
    Analyze consolidation benefits and determine optimal strategy
    
    Consolidation Logic:
    1. Collect all shipments at a central hub (usually nearest port city)
    2. Combine compatible commodities
    3. Optimize container loading
    4. Calculate volume vs weight efficiency
    5. Apply consolidation discounts
    
    Assumptions:
    - Minimum 2 tons total weight for consolidation benefits
    - Compatible commodities can be mixed (no hazardous with food)
    - 85% volume efficiency due to better packing
    - 5% shipping discount for consolidated loads
    - 2-day consolidation time
    """
    
    total_weight = sum(supplier.total_weight for supplier in suppliers)
    total_volume = sum(supplier.total_volume for supplier in suppliers)
    total_value = sum(supplier.total_value for supplier in suppliers)
    
    # Check commodity compatibility
    commodity_types = set()
    for supplier in suppliers:
        commodity_types.add(supplier.get_dominant_commodity())
    
    # Incompatible combinations
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
    
    # Calculate consolidation benefits
    consolidation_feasible = (
        total_weight >= PRICING_CONFIG["consolidation_factors"]["min_consolidation_weight"] and
        is_compatible and
        len(suppliers) >= 2
    )
    
    if consolidation_feasible:
        # Volume efficiency improvement
        optimized_volume = total_volume * PRICING_CONFIG["consolidation_factors"]["volume_efficiency_factor"]
        
        # Consolidation costs
        handling_fees = len(suppliers) * PRICING_CONFIG["consolidation_factors"]["handling_fee_per_supplier"]
        warehouse_fees = (PRICING_CONFIG["consolidation_factors"]["consolidation_time_days"] * 
                         PRICING_CONFIG["consolidation_factors"]["warehouse_fee_per_day"])
        
        consolidation_costs = handling_fees + warehouse_fees
        
        # Savings from consolidation
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
            "incompatible_reason": "Minimum weight not met" if total_weight < 2.0 else "Incompatible commodities",
            "consolidation_time_days": 0
        }

def get_optimal_port(suppliers: List[Supplier], destination_country: str, 
                    consolidation_analysis: Dict) -> Dict:
    """Find optimal port considering all supplier locations and consolidation"""
    
    port_analysis = {}
    
    for port_city, info in WORLDREF_HUBS.items():
        if port_city == "Delhi":
            continue
        
        total_domestic_cost = 0
        max_distance = 0
        
        # Calculate cost from each supplier to port
        for supplier in suppliers:
            distance, duration = calculate_distance_with_ai(supplier.location, port_city)
            weight_factor = get_weight_factor(supplier.total_weight)
            supplier_cost = (distance * PRICING_CONFIG["base_rate_per_km_per_ton"] * 
                           weight_factor * supplier.total_weight)
            
            # Apply commodity factor
            commodity_factor = PRICING_CONFIG["commodity_factors"].get(
                supplier.get_dominant_commodity(), 1.0
            )
            supplier_cost *= commodity_factor
            
            total_domestic_cost += supplier_cost
            max_distance = max(max_distance, distance)
        
        # Add consolidation costs if applicable
        if consolidation_analysis["feasible"]:
            total_domestic_cost += consolidation_analysis["consolidation_costs"]
        
        # Port charges
        port_charges = PRICING_CONFIG["port_charges"].get(port_city, 9000)
        
        # Specialization bonus
        specialization_bonus = 0.9 if destination_country in info["specialization"] else 1.0
        
        # Apply consolidation discount to shipping
        total_cost = (total_domestic_cost + port_charges) * specialization_bonus
        
        port_analysis[port_city] = {
            "total_domestic_cost": total_domestic_cost,
            "max_distance_km": max_distance,
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
                                        consolidation_analysis: Dict) -> float:
    """
    Calculate international shipping cost based on weight/volume and consolidation
    
    Assumptions:
    - Shipping rates: $2000/ton or $300/CBM, whichever is higher
    - Consolidation provides 5% discount
    - Charges only till destination country port (no inland delivery)
    """
    
    # Base rates
    weight_based_rate = 2000  # USD per ton
    volume_based_rate = 300   # USD per CBM
    
    # Use optimized volume if consolidated
    effective_volume = consolidation_analysis["optimized_volume"]
    
    # Calculate costs
    weight_cost = total_weight * weight_based_rate
    volume_cost = effective_volume * volume_based_rate
    
    # Take higher of weight or volume cost
    base_international_cost = max(weight_cost, volume_cost)
    
    # Apply consolidation discount
    international_cost = base_international_cost * consolidation_analysis["shipping_discount"]
    
    return international_cost

def calculate_multi_supplier_cost(suppliers: List[Supplier], destination_country: str, 
                                destination_port: str, service_level: str) -> Dict:
    """Calculate total cost for multi-supplier shipment with consolidation"""
    
    # Analyze consolidation benefits
    consolidation_analysis = analyze_consolidation_benefits(suppliers, "Mumbai")  # Default hub
    
    # Get optimal port
    port_result = get_optimal_port(suppliers, destination_country, consolidation_analysis)
    optimal_port = port_result["port"]
    port_analysis = port_result["analysis"]
    
    # Calculate costs
    domestic_cost = port_analysis[optimal_port]["total_domestic_cost"]
    port_charges = port_analysis[optimal_port]["port_charges"]
    
    # International shipping (till destination port only)
    international_cost = calculate_international_shipping_cost(
        consolidation_analysis["total_weight"],
        consolidation_analysis["total_volume"],
        destination_country,
        consolidation_analysis
    )
    
    # Partner service cost
    partner_rate = PRICING_CONFIG["partner_rates"][service_level]
    partner_cost = partner_rate * consolidation_analysis["total_weight"]
    
    # Insurance (2% of cargo value)
    insurance_cost = consolidation_analysis["total_value"] * 0.02
    
    # Documentation (fixed cost)
    documentation_cost = 5000
    
    # Total cost
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
            "documentation": documentation_cost,
            "consolidation_costs": consolidation_analysis["consolidation_costs"]
        },
        "consolidation_analysis": consolidation_analysis,
        "optimal_port": optimal_port,
        "port_analysis": port_analysis,
        "suppliers_summary": {
            "count": len(suppliers),
            "total_weight": consolidation_analysis["total_weight"],
            "total_volume": consolidation_analysis["total_volume"],
            "total_value": consolidation_analysis["total_value"]
        }
    }

def main():
    st.set_page_config(
        page_title="WorldRef Multi-Supplier Consolidation Estimator",
        page_icon="📦",
        layout="wide"
    )
    
    st.title("📦 WorldRef Multi-Supplier Consolidation Estimator")
    st.markdown("Advanced transport cost calculation with multi-supplier consolidation and optimization")
    
    # Initialize session state
    if 'suppliers' not in st.session_state:
        st.session_state.suppliers = []
    
    # Sidebar for shipment details
    st.sidebar.header("📋 Shipment Details")
    
    # Destination
    destination_country = st.sidebar.selectbox("Destination Country", DESTINATION_COUNTRIES, index=0)
    destination_port = st.sidebar.text_input("Destination Port", value="Main Port")
    service_level = st.sidebar.selectbox("Service Level", ["Express", "Standard", "Economy"], index=1)
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📦 Supplier Management", "💰 Cost Analysis", "📊 Consolidation Report"])
    
    with tab1:
        st.header("Supplier and Packing List Management")
        
        # Add new supplier
        st.subheader("Add New Supplier")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            supplier_name = st.text_input("Supplier Name")
        with col2:
            supplier_location = st.selectbox("Supplier Location", INDIAN_CITIES)
        with col3:
            supplier_contact = st.text_input("Contact Information")
        
        if st.button("Add Supplier"):
            if supplier_name and supplier_name.strip() and supplier_location:
                new_supplier = Supplier(supplier_name.strip(), supplier_location, supplier_contact)
                st.session_state.suppliers.append(new_supplier)
                st.success(f"Supplier {supplier_name} added successfully!")
                st.rerun()
            else:
                st.error("Please fill in supplier name and location")
        
        # Display and manage existing suppliers
        if st.session_state.suppliers:
            st.subheader("Existing Suppliers")
            
            for i, supplier in enumerate(st.session_state.suppliers):
                with st.expander(f"📍 {supplier.name} - {supplier.location}"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**Contact:** {supplier.contact}")
                        st.write(f"**Total Weight:** {supplier.total_weight:.2f} tons")
                        st.write(f"**Total Volume:** {supplier.total_volume:.2f} CBM")
                        st.write(f"**Total Value:** ${supplier.total_value:,.2f}")
                    
                    with col2:
                        if st.button(f"Remove {supplier.name}", key=f"remove_{i}"):
                            st.session_state.suppliers.pop(i)
                            st.rerun()
                    
                    # Add packing items
                    st.write("**Add Packing Items:**")
                    
                    with st.form(key=f"item_form_{i}"):
                        pcol1, pcol2, pcol3 = st.columns(3)
                        
                        with pcol1:
                            item_desc = st.text_input("Item Description", key=f"desc_{i}")
                            item_qty = st.number_input("Quantity", min_value=1, value=1, key=f"qty_{i}")
                            item_weight = st.number_input("Weight per Unit (kg)", min_value=0.1, value=1.0, step=0.1, key=f"weight_{i}")
                        
                        with pcol2:
                            item_volume = st.number_input("Volume per Unit (CBM)", min_value=0.01, value=0.1, step=0.01, key=f"volume_{i}")
                            item_value = st.number_input("Value per Unit ($)", min_value=0.01, value=10.0, step=0.01, key=f"value_{i}")
                            item_commodity = st.selectbox("Commodity Type", 
                                                        list(PRICING_CONFIG["commodity_factors"].keys()),
                                                        key=f"commodity_{i}")
                        
                        with pcol3:
                            st.write("")  # Empty space for alignment
                            st.write("")  # Empty space for alignment
                            add_item_button = st.form_submit_button(f"Add Item to {supplier.name}")
                        
                        if add_item_button:
                            if item_desc and item_desc.strip():
                                new_item = PackingItem(
                                    item_desc, item_qty, item_weight/1000,  # Convert kg to tons
                                    item_volume, item_commodity, item_value
                                )
                                supplier.add_packing_item(new_item)
                                st.success(f"Item '{item_desc}' added to {supplier.name}'s packing list!")
                                st.rerun()
                            else:
                                st.error("Please enter item description")
                    
                    # Display packing list
                    if supplier.packing_list:
                        st.write("**Current Packing List:**")
                        packing_df = pd.DataFrame([
                            {
                                "Description": item.description,
                                "Quantity": item.quantity,
                                "Unit Weight (kg)": f"{item.weight_per_unit * 1000:.2f}",
                                "Unit Volume (CBM)": f"{item.volume_per_unit:.3f}",
                                "Commodity": item.commodity_type,
                                "Unit Value ($)": f"{item.value:.2f}",
                                "Total Weight (kg)": f"{item.total_weight * 1000:.2f}",
                                "Total Volume (CBM)": f"{item.total_volume:.3f}",
                                "Total Value ($)": f"{item.value * item.quantity:.2f}"
                            }
                            for item in supplier.packing_list
                        ])
                        st.dataframe(packing_df, use_container_width=True)
                        
                        # Option to clear packing list
                        if st.button(f"Clear Packing List", key=f"clear_{i}"):
                            supplier.packing_list = []
                            supplier.total_weight = 0
                            supplier.total_volume = 0
                            supplier.total_value = 0
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
                        destination_port, service_level
                    )
                    
                    # Display key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Cost", f"₹{cost_data['total_cost']:,.0f}", 
                                f"~${cost_data['total_cost']/83:,.0f} USD")
                    
                    with col2:
                        st.metric("Total Weight", f"{cost_data['suppliers_summary']['total_weight']:.2f} tons")
                    
                    with col3:
                        st.metric("Total Volume", f"{cost_data['suppliers_summary']['total_volume']:.2f} CBM")
                    
                    with col4:
                        st.metric("Optimal Port", cost_data['optimal_port'])
                    
                    # Cost breakdown
                    st.subheader("📊 Cost Breakdown")
                    
                    breakdown_data = cost_data['breakdown']
                    fig = px.pie(
                        values=list(breakdown_data.values()),
                        names=[name.replace("_", " ").title() for name in breakdown_data.keys()],
                        title="Cost Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed breakdown table
                    breakdown_df = pd.DataFrame([
                        {"Component": k.replace("_", " ").title(), "Cost (₹)": f"{v:,.0f}"}
                        for k, v in breakdown_data.items()
                    ])
                    st.dataframe(breakdown_df, use_container_width=True)
                    
                    # Consolidation analysis
                    st.subheader("🔄 Consolidation Analysis")
                    
                    cons_analysis = cost_data['consolidation_analysis']
                    
                    if cons_analysis['feasible']:
                        st.success("✅ Consolidation is feasible and beneficial!")
                        
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
                    
                    # Port analysis
                    st.subheader("🚢 Port Analysis")
                    port_df = pd.DataFrame([
                        {
                            "Port": port,
                            "Total Domestic Cost (₹)": f"{data['total_domestic_cost']:,.0f}",
                            "Port Charges (₹)": f"{data['port_charges']:,.0f}",
                            "Total Cost (₹)": f"{data['total_cost']:,.0f}",
                            "Suitable": "✅" if data['suitable'] else "❌"
                        }
                        for port, data in cost_data['port_analysis'].items()
                    ])
                    st.dataframe(port_df, use_container_width=True)
        else:
            st.info("Please add at least one supplier to calculate costs.")
    
    with tab3:
        st.header("📊 Consolidation Report")
        
        if len(st.session_state.suppliers) >= 1:
            # Supplier summary
            st.subheader("Supplier Summary")
            
            summary_data = []
            for supplier in st.session_state.suppliers:
                summary_data.append({
                    "Supplier": supplier.name,
                    "Location": supplier.location,
                    "Items": len(supplier.packing_list),
                    "Weight (tons)": f"{supplier.total_weight:.2f}",
                    "Volume (CBM)": f"{supplier.total_volume:.2f}",
                    "Value ($)": f"{supplier.total_value:,.2f}",
                    "Dominant Commodity": supplier.get_dominant_commodity()
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
            # Consolidation assumptions
            st.subheader("📋 Consolidation Logic & Assumptions")
            
            st.markdown("""
            **Consolidation Benefits Logic:**
            1. **Minimum Requirements:** At least 2 tons total weight and 2+ suppliers
            2. **Commodity Compatibility:** Hazardous materials cannot be mixed with food/perishables
            3. **Volume Optimization:** 15% volume efficiency gain through better packing
            4. **Shipping Discount:** 5% discount on international shipping for consolidated loads
            5. **Consolidation Hub:** Nearest port city for optimal cost efficiency
            
            **Cost Assumptions:**
            - **Domestic Transport:** ₹15 per km per ton with weight-based multipliers
            - **Handling Fee:** ₹2,000 per supplier for consolidation
            - **Warehouse Storage:** ₹500 per day (2-day consolidation time)
            - **International Shipping:** Higher of $2,000/ton or $300/CBM
            - **Insurance:** 2% of total cargo value
            - **Coverage:** Costs calculated till destination country port only
            
            **Port Selection Criteria:**
            - Distance from suppliers to port
            - Port specialization for destination region
            - Total cost optimization including consolidation
            """)
        else:
            st.info("Add suppliers to view consolidation report.")

if __name__ == "__main__":
    main()
