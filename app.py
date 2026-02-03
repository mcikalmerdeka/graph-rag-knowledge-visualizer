"""
Knowledge Graph Generator with Graph RAG

A Streamlit application for:
1. Generating knowledge graphs from uploaded documents
2. Querying the knowledge graph using natural language with Graph RAG
"""

import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
import random
import string

# Import our modules
from src.main import KnowledgeGraphGenerator
from src.core.visualization import GraphVisualizer
from src.core.graph_rag import GraphRAG, create_graph_rag


def generate_random_hash(length=4):
    """Generate a random hash string of specified length."""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))


def initialize_session_state():
    """Initialize session state variables."""
    if 'graph_generated' not in st.session_state:
        st.session_state.graph_generated = False
    if 'graph_html_path' not in st.session_state:
        st.session_state.graph_html_path = None
    if 'graph_rag' not in st.session_state:
        st.session_state.graph_rag = None
    if 'graph_document' not in st.session_state:
        st.session_state.graph_document = None
    if 'text_content' not in st.session_state:
        st.session_state.text_content = ""
    if 'example_loaded' not in st.session_state:
        st.session_state.example_loaded = False


def get_example_files():
    """Get list of example files from data folder."""
    data_dir = Path("data")
    if not data_dir.exists():
        return []
    return sorted([f.name for f in data_dir.glob("*.txt")])


def load_example_file(filename):
    """Load content of an example file."""
    file_path = Path("data") / filename
    if file_path.exists():
        return file_path.read_text(encoding="utf-8")
    return ""


def load_selected_example():
    """Callback to load the selected example content directly into session state."""
    selected = st.session_state.example_selector
    if selected and selected != "Select an example...":
        # We need to find the filename from the display name
        # This repeats some logic but is necessary inside the callback
        files = get_example_files()
        display_names = [f.replace("_", " ").replace(".txt", "").title() for f in files]
        
        try:
            index = display_names.index(selected)
            filename = files[index]
            content = load_example_file(filename)
            if content:
                st.session_state.text_content = content
                st.session_state.text_input_area = content  # Sync widget state
                st.session_state.example_loaded = True
        except ValueError:
            pass


def clear_example_selection():
    """Callback to clear the example selection."""
    st.session_state.text_content = ""
    st.session_state.text_input_area = ""  # Sync widget state
    # resetting the widget key in session state resets the widget
    st.session_state.example_selector = "Select an example..."
    st.session_state.example_loaded = False

def render_graph_generation_section():
    """Render the knowledge graph generation section."""
    st.header("📄 1. Generate Knowledge Graph")
    st.markdown("Upload a document, paste text, or select an example from the sidebar to extract a knowledge graph.")
    
    # Input method selection - default to "Paste text directly" if content exists
    default_index = 1 if st.session_state.text_content else 0
    
    input_method = st.radio(
        "Choose input method:",
        ["Upload .txt file", "Paste text directly"],
        horizontal=True,
        index=default_index,
        key="input_method_radio"
    )
    
    text_content = None
    
    if input_method == "Upload .txt file":
        uploaded_file = st.file_uploader(
            "Upload a text file",
            type=["txt"],
            help="Upload a .txt file to generate knowledge graph",
        )
        
        if uploaded_file is not None:
            text_content = uploaded_file.read().decode("utf-8")
            st.success(f"✅ File uploaded: {uploaded_file.name} ({len(text_content)} characters)")
    
    elif input_method == "Paste text directly":
        text_content = st.text_area(
            "Paste your text here",
            value=st.session_state.text_content,
            height=200,
            placeholder="Enter the text you want to analyze...",
            key="text_input_area",
        )
        
        # Update session state when text area changes
        if text_content != st.session_state.text_content:
            st.session_state.text_content = text_content
        
        if text_content:
            st.success(f"✅ Text entered: {len(text_content)} characters")
    
    # Generate button
    if text_content and st.button("🔮 Generate Knowledge Graph", type="primary", use_container_width=True):
        with st.spinner("Extracting knowledge graph using AI... This may take a moment."):
            try:
                # Generate knowledge graph using the new API
                generator = KnowledgeGraphGenerator()
                graphs = generator.generate_sync(text_content)
                
                if graphs:
                    # Create visualizer and generate HTML
                    visualizer = GraphVisualizer()
                    
                    # Ensure output directory exists
                    output_dir = Path("output")
                    output_dir.mkdir(exist_ok=True)
                    
                    # Generate filename with random hash
                    random_hash = generate_random_hash(4)
                    output_file = output_dir / f"knowledge_graph_{random_hash}.html"
                    visualizer.visualize(graphs[0], output_file=output_file)
                    
                    # Store graph document and create RAG instance
                    st.session_state.graph_document = graphs[0]
                    st.session_state.graph_rag = create_graph_rag(graphs[0])
                    st.session_state.graph_html_path = str(output_file)
                    st.session_state.graph_generated = True
                    
                    # Get stats
                    stats = generator.get_stats(graphs)
                    
                    st.success("✅ Knowledge graph generated successfully!")
                    
                    # Display stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Nodes", stats['total_nodes'])
                    with col2:
                        st.metric("Relationships", stats['total_relationships'])
                    with col3:
                        st.metric("Node Types", stats['unique_node_types'])
                    
                    st.info("👇 Now you can ask questions about this knowledge graph in Section 2 below!")
                    
                else:
                    st.error("❌ Failed to generate knowledge graph.")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)
    
    # Display graph if already generated (persists across interactions)
    if st.session_state.graph_generated and st.session_state.graph_html_path:
        st.subheader("📊 Knowledge Graph Visualization")
        with open(st.session_state.graph_html_path, 'r', encoding='utf-8') as f:
            components.html(f.read(), height=800, scrolling=True)


def render_rag_section():
    """Render the Graph RAG Q&A section."""
    st.header("💬 2. Ask Questions (Graph RAG)")
    
    if not st.session_state.graph_generated or st.session_state.graph_rag is None:
        st.info("⬆️ Please generate a knowledge graph first in Section 1 above.")
        return
    
    # Show graph stats
    stats = st.session_state.graph_rag.get_stats()
    
    with st.expander("📊 Knowledge Graph Statistics", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Nodes", stats['total_nodes'])
        with col2:
            st.metric("Total Edges", stats['total_edges'])
        with col3:
            st.metric("Graph Density", f"{stats['density']:.3f}")
        
        # Show node type distribution
        if stats['node_types']:
            st.write("**Node Types:**")
            for node_type, count in sorted(stats['node_types'].items(), key=lambda x: x[1], reverse=True):
                st.write(f"  - {node_type}: {count}")
    
    # Query input
    st.markdown("---")
    st.subheader("Ask a Question")
    
    question = st.text_input(
        "Enter your question about the knowledge graph:",
        placeholder="e.g., What are the side effects of metformin? Who is the CEO of Tesla?",
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)
    with col2:
        show_context = st.checkbox("Show retrieved context", value=False)
    
    if ask_button and question:
        with st.spinner("Searching knowledge graph and generating answer..."):
            try:
                # Query the graph
                result = st.session_state.graph_rag.query(question)
                
                # Display answer
                st.markdown("---")
                st.subheader("📝 Answer")
                st.markdown(result['answer'])
                
                # Display confidence
                confidence = result.get('confidence', 0)
                if confidence > 0:
                    st.progress(confidence, text=f"Confidence: {confidence:.0%}")
                
                # Display context if requested
                if show_context and result.get('context_text'):
                    with st.expander("🔍 Retrieved Context"):
                        st.text(result['context_text'])
                        
                        # Show relevant entities
                        if result.get('sources'):
                            st.write("**Relevant Entities:**")
                            for entity in result['sources'][:10]:
                                st.write(f"  - {entity}")
                
            except Exception as e:
                st.error(f"❌ Error querying knowledge graph: {str(e)}")
                st.exception(e)
    
def main():
    """Main Streamlit application."""
    # Page configuration
    st.set_page_config(
        page_title="Knowledge Graph Generator & RAG",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar - Example Files
    with st.sidebar:
        st.header("📚 Example Files")
        st.markdown("Select an example to auto-fill the text area.")
        
        example_files = get_example_files()
        if example_files:
            # Create a selectbox with formatted names
            example_options = ["Select an example..."] + [
                f.replace("_", " ").replace(".txt", "").title() 
                for f in example_files
            ]
            
            # Selectbox with callback
            st.selectbox(
                "Choose an example:",
                options=example_options,
                key="example_selector",
                on_change=load_selected_example
            )
            
            # Show success message if example is loaded
            if st.session_state.get('example_selector') != "Select an example..." and st.session_state.text_content:
                st.success(f"✅ Loaded: {st.session_state.example_selector}")
                st.info(f"📄 {len(st.session_state.text_content)} characters")
                
                # Add a button to clear the selection
                st.button(
                    "🗑️ Clear Selection", 
                    use_container_width=True,
                    on_click=clear_example_selection
                )
        else:
            st.warning("No example files found in data/ folder.")
        
        st.markdown("---")
        st.markdown("### 📁 Available Examples")
        if example_files:
            for i, f in enumerate(example_files, 1):
                display_name = f.replace("_", " ").replace(".txt", "").title()
                st.write(f"{i}. {display_name}")
    
    # Title and description
    st.markdown("<h1 style='text-align: center;'>🧠 Knowledge Graph Generator with Graph RAG</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'><strong>Transform unstructured text into interactive knowledge graphs and query them using AI.</strong></p>", unsafe_allow_html=True)
    
    # Application description in expander
    with st.expander("ℹ️ About this application", expanded=False):
        st.markdown("""
        This application uses Large Language Models to:
        1. **Extract structured knowledge** from your documents
        2. **Create interactive visualizations** of entities and relationships  
        3. **Answer questions** using Graph RAG (Retrieval-Augmented Generation)
        """)
    
    st.markdown("---")
    
    # Main sections
    render_graph_generation_section()
    
    st.markdown("---")
    
    render_rag_section()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        Built with LangChain + NetworkX + OpenAI + PyVis
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
