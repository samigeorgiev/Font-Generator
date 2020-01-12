import React from 'react';

import styles from './index.module.css';

const backdrop = props => (
    props.isShown ? <div className={styles.Backdrop} onClick={props.click} /> : null
);

export default backdrop;