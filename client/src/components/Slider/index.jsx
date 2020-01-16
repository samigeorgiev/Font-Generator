import React from 'react';

const slider = props => {
    return (
        <input type="range" min="1" max="10" onChange={props.change} value={props.value} />
    );
};

export default slider;