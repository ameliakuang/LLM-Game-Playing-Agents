#!/usr/bin/env python3
import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set session information
SESSION_ID = "20250323_115142"
FRAME_NUM = 46
GAME_TYPE = "breakout"  # Can be "pong", "space_invaders", or "breakout"

# Define paths for the session data
def get_session_paths(session_id, frame_num):
    """Get paths to the raw frame, visualization frame, and state data for a specific frame."""
    base_dir = Path("my_gameplay") / session_id
    
    # Create file paths
    raw_frame_path = base_dir / "raw_frames" / f"frame_{frame_num:06d}.png"
    vis_frame_path = base_dir / "vis_frames" / f"vis_{frame_num:06d}.png"
    state_path = base_dir / "states" / f"state_{frame_num:06d}.json"
    
    return raw_frame_path, vis_frame_path, state_path

def load_data(raw_frame_path, vis_frame_path, state_path):
    """Load the raw frame, visualization frame, and state data."""
    # Check if files exist
    if not vis_frame_path.exists():
        raise FileNotFoundError(f"Visualization frame not found: {vis_frame_path}")
    
    if not state_path.exists():
        raise FileNotFoundError(f"State data not found: {state_path}")
    
    # Load raw frame if it exists
    raw_frame = None
    if raw_frame_path.exists():
        raw_frame = cv2.imread(str(raw_frame_path))
        # Convert BGR to RGB for matplotlib
        raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
    
    # Load the visualization frame
    vis_frame = cv2.imread(str(vis_frame_path))
    vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
    
    # Load the state data
    with open(state_path, 'r') as f:
        state_data = json.load(f)
    
    return raw_frame, vis_frame, state_data

# Create draggable annotation class
class DraggableAnnotation:
    def __init__(self, annotation, line=None):
        self.annotation = annotation
        self.line = line
        self.press = None
        self.background = None
        self.connect()
        
    def connect(self):
        """Connect to all the events we need."""
        self.cidpress = self.annotation.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.annotation.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.annotation.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)
    
    def on_press(self, event):
        """Called when a mouse button is pressed over annotation."""
        if event.inaxes != self.annotation.axes:
            return
        contains, attrd = self.annotation.contains(event)
        if not contains:
            return
        x0, y0 = self.annotation.get_position()
        self.press = x0, y0, event.xdata, event.ydata
        
    def on_motion(self, event):
        """Called during mouse movement."""
        if self.press is None:
            return
        if event.inaxes != self.annotation.axes:
            return
        x0, y0, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.annotation.set_position((x0+dx, y0+dy))
        
        # Update line if it exists
        if self.line is not None:
            # Get current line data
            xdata, ydata = self.line.get_data()
            # Update the endpoint of the line to match the annotation position
            xdata[-1] = x0 + dx
            ydata[-1] = y0 + dy
            self.line.set_data(xdata, ydata)
        
        self.annotation.figure.canvas.draw_idle()
        
    def on_release(self, event):
        """Called when the mouse button is released."""
        self.press = None
        self.annotation.figure.canvas.draw_idle()
        
    def disconnect(self):
        """Disconnect all callbacks."""
        self.annotation.figure.canvas.mpl_disconnect(self.cidpress)
        self.annotation.figure.canvas.mpl_disconnect(self.cidrelease)
        self.annotation.figure.canvas.mpl_disconnect(self.cidmotion)

def get_block_color(y):
    """Determine block color based on y-coordinate for Breakout."""
    if y == 57: return (0/255, 0/255, 1.0)    # Blue
    elif y == 63: return (0/255, 1.0, 1.0)    # Aqua
    elif y == 69: return (0/255, 1.0, 0/255)  # Green
    elif y == 75: return (1.0, 1.0, 0/255)    # Yellow
    elif y == 81: return (1.0, 165/255, 0/255)  # Orange
    elif y == 87: return (1.0, 0/255, 0/255)  # Red
    else: return (1.0, 1.0, 1.0)              # Default white

def plot_side_by_side(raw_frame, vis_frame, state_data, output_path=None):
    """Create a side-by-side plot with raw/vis frame and detailed data labels."""
    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Determine what to show on the left
    if raw_frame is not None:
        ax1.imshow(raw_frame)
        ax1.set_title("Raw Game Frame")
    else:
        ax1.imshow(vis_frame)
        ax1.set_title("Visualization Frame")
    ax1.axis('off')
    
    # Display visualization frame on the right with detailed annotations
    ax2.imshow(vis_frame)
    ax2.set_title("Annotated with State Data")
    ax2.axis('off')
    
    # Define objects to highlight based on game type
    if GAME_TYPE == "pong":
        objects_to_highlight = ['Player', 'Enemy', 'Ball']
        # Define colors for Pong objects (normalized for matplotlib)
        colors = {
            'Player': (0/255, 1.0, 0/255),    # Green for player paddle
            'Enemy': (0/255, 0/255, 1.0),     # Dark blue for enemy paddle
            'Ball': (0/255, 1.0, 1.0),        # Cyan for ball
        }
    elif GAME_TYPE == "breakout":
        objects_to_highlight = ['Player', 'Ball']  # We'll handle blocks separately
        # Define colors for Breakout objects
        colors = {
            'Player': (0/255, 1.0, 0/255),    # Green for player paddle
            'Ball': (0/255, 1.0, 1.0),        # Cyan for ball
        }
    else:  # space_invaders
        objects_to_highlight = [
            'Player', 
            'Shield0', 'Shield1', 'Shield2',  # All 3 shields
            'Bullet0', 'Bullet1',             # Both bullets
            'Alien0', 'Alien5', 'Alien10'     # Selected aliens
        ]
        # Define colors for Space Invaders objects
        colors = {
            'Player': (0/255, 1.0, 0/255),     # Green
            'Shield': (1.0, 0/255, 0/255),     # Red 
            'Alien': (0/255, 0/255, 1.0),      # Blue
            'Bullet': (0/255, 1.0, 1.0),       # Cyan
        }
    
    # Store draggable annotations for interactivity
    draggable_annotations = []
    
    # Add text labels for selected objects with detailed state data
    for key, obj in state_data.items():
        # Skip if not a dictionary (like reward)
        if not isinstance(obj, dict):
            continue
            
        # For Breakout, handle blocks separately
        if GAME_TYPE == "breakout" and key.startswith('Block'):
            objects_to_highlight.append(key)
            continue
            
        # Skip if not in our highlight list
        if key not in objects_to_highlight:
            continue
            
        try:
            # Extract position and velocity data
            x = float(obj.get('x', 0))
            y = float(obj.get('y', 0))
            w = float(obj.get('w', 5))
            h = float(obj.get('h', 5))
            dx = float(obj.get('dx', 0))
            dy = float(obj.get('dy', 0))
            
            # Determine color based on game type and object
            if GAME_TYPE == "pong":
                color = colors.get(key, (1.0, 1.0, 1.0))  # White for unknown objects
            elif GAME_TYPE == "breakout":
                if key.startswith('Block'):
                    color = get_block_color(y)
                else:
                    color = colors.get(key, (1.0, 1.0, 1.0))
            else:
                if key == 'Player':
                    color = colors['Player']
                elif key.startswith('Shield'):
                    color = colors['Shield']
                elif key.startswith('Alien'):
                    color = colors['Alien']
                elif key.startswith('Bullet'):
                    color = colors['Bullet']
                else:
                    color = (1.0, 1.0, 1.0)  # White for other objects
            
            # Create simplified state info label with larger text
            state_text = f"pos=({int(x)},{int(y)})\nvel=({dx:.1f},{dy:.1f})"
            
            # Position label based on game type and object
            if GAME_TYPE == "pong":
                if key == 'Player':
                    # Position label to the right of the player paddle
                    x_offset = w + 2
                    y_offset = 0
                elif key == 'Enemy':
                    # Position label to the left of the enemy paddle
                    x_offset = -20
                    y_offset = 0
                else:  # Ball
                    # Position label above the ball
                    x_offset = 0
                    y_offset = -10
            elif GAME_TYPE == "breakout":
                if key == 'Player':
                    # Position label to the right of the player paddle
                    x_offset = w + 2
                    y_offset = 0
                elif key == 'Ball':
                    # Position label above the ball
                    x_offset = 0
                    y_offset = -10
                elif key.startswith('Block'):
                    # For blocks, position label above them
                    x_offset = 0
                    y_offset = -5
            else:
                # Space Invaders positioning logic
                x_offset = w + 2
                y_offset = 0
                
                if key.startswith('Alien'):
                    if key == 'Alien0':
                        y_offset = -5
                    elif key == 'Alien5':
                        y_offset = 0
                    elif key == 'Alien10':
                        y_offset = 5
                elif key.startswith('Shield'):
                    if key == 'Shield0':
                        x_offset += 5
                    elif key == 'Shield1':
                        x_offset += 15
                    elif key == 'Shield2':
                        x_offset += 25
                elif key.startswith('Bullet'):
                    if key == 'Bullet0':
                        x_offset += 5
                    elif key == 'Bullet1':
                        x_offset += 15
            
            # Add detailed label with position and velocity data
            annotation = ax2.annotate(
                state_text,
                (x + x_offset, y + y_offset),
                color=color,
                fontsize=10,
                weight='bold',
                backgroundcolor='black',
                va='center',
                bbox=dict(boxstyle="round,pad=0.3", fc='black', ec=color, alpha=0.8)
            )
            
            # Draw a line to connect the object to its label
            line = ax2.plot(
                [x + w/2, x + x_offset - 2],
                [y + h/2, y + y_offset],
                color=color,
                linestyle='-',
                linewidth=1
            )[0]
            
            # Make annotation draggable
            draggable = DraggableAnnotation(annotation, line)
            draggable_annotations.append(draggable)
            
        except (ValueError, TypeError) as e:
            print(f"Error annotating {key}: {e}")
    
    # Handle Breakout blocks separately
    if GAME_TYPE == "breakout":
        for key, obj in state_data.items():
            if not isinstance(obj, dict) or not key.startswith('Block'):
                continue
                
            try:
                # Extract position and velocity data
                x = float(obj.get('x', 0))
                y = float(obj.get('y', 0))
                w = float(obj.get('w', 5))
                h = float(obj.get('h', 5))
                dx = float(obj.get('dx', 0))
                dy = float(obj.get('dy', 0))
                
                # Get color based on y-coordinate
                color = get_block_color(y)
                
                # Create simplified state info label
                state_text = f"pos=({int(x)},{int(y)})\nvel=({dx:.1f},{dy:.1f})"
                
                # Position label above the block
                x_offset = 0
                y_offset = -5
                
                # Add detailed label with position and velocity data
                annotation = ax2.annotate(
                    state_text,
                    (x + x_offset, y + y_offset),
                    color=color,
                    fontsize=10,
                    weight='bold',
                    backgroundcolor='black',
                    va='center',
                    bbox=dict(boxstyle="round,pad=0.3", fc='black', ec=color, alpha=0.8)
                )
                
                # Draw a line to connect the block to its label
                line = ax2.plot(
                    [x + w/2, x + x_offset - 2],
                    [y + h/2, y + y_offset],
                    color=color,
                    linestyle='-',
                    linewidth=1
                )[0]
                
                # Make annotation draggable
                draggable = DraggableAnnotation(annotation, line)
                draggable_annotations.append(draggable)
                
            except (ValueError, TypeError) as e:
                print(f"Error annotating {key}: {e}")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure if an output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
        print("Note: The saved image will not have interactive features.")
    
    # Add instruction about interactivity
    print("\nInteractive features:")
    print("- Click and drag any label to reposition it")
    print("- The connecting line will follow the label")
    print("- Close the plot window when finished viewing")
    
    # Display the plot (this will be interactive)
    plt.show()

def main():
    try:
        # Get file paths
        raw_frame_path, vis_frame_path, state_path = get_session_paths(SESSION_ID, FRAME_NUM)
        
        # Load data
        raw_frame, vis_frame, state_data = load_data(raw_frame_path, vis_frame_path, state_path)
        
        # Create the side-by-side plot
        output_path = f"frame_{FRAME_NUM}_analysis.png"
        plot_side_by_side(raw_frame, vis_frame, state_data, output_path)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        # Try to help the user find available sessions
        try:
            gameplay_dir = Path("my_gameplay")
            if gameplay_dir.exists():
                print("\nAvailable sessions:")
                for session_dir in gameplay_dir.iterdir():
                    if session_dir.is_dir():
                        print(f"- {session_dir.name}")
        except Exception:
            pass
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 