import React from 'react';

import styles from './index.module.css';

const textArea = props => (
    <textarea
        className={styles.TextArea}
        defaultValue={props.children}
        style={{
            fontFamily: props.font,
            fontSize: props.size
        }}
    />
);

export default textArea;