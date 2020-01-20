import React from 'react';

import styles from './index.module.css';

const slider = props => {
    return (
        <input
            type="range"
            min={props.min}
            max={props.max}
            step={props.step}
            className={styles.Slider}
            value={props.value}
            onChange={props.change}
        />
    );
};

export default slider;